import pandas as pd
import numpy as np
import warnings
import os
from typing import NamedTuple

# ignore FutureWarning: DataFrameGroupBy.apply
warnings.filterwarnings("ignore", category=FutureWarning)

class ProcessedData(NamedTuple):
    comments: pd.DataFrame
    replies: pd.DataFrame
    user_pairs: pd.DataFrame
    submissions: pd.DataFrame

class DataProcessor:
    def __init__(self, base_configs: dict):
        self.base_configs = base_configs

        # Paths
        self.raw_path = self.base_configs['paths']['raw']
        self.processed_path = self.base_configs['paths']['processed']

        # Maps
        self.labels_map = self.base_configs['labels']
        self.subreddits_map = self.base_configs['subreddits']
        self.required_columns = self.base_configs['required_columns']
        
        # Processing configs
        self.cleaning_cfgs = self.base_configs['cleaning']
        self.temporal_cfgs = self.base_configs['temporal']
        self.default_values = self.base_configs['default_values']
        self.embedding_cfgs = self.base_configs['text_embedding']

    def clean_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        processed = raw_data.copy()
        # 1. Check if mandatory columns exist
        missing_cols = set(self.required_columns) - set(processed.columns)
        if missing_cols:
            raise ValueError(f"Missing mandatory columns: {missing_cols}")

        #=======================================
        # CLEANING
        #=======================================
        # 2. Normalize subreddit names
        if self.cleaning_cfgs.get('normalize_subreddits', False):
            processed['subreddit'] = processed['subreddit'].str.lower()

        # 3. Rename columns
        if 'rename_columns' in self.cleaning_cfgs:
            processed = processed.rename(columns=self.cleaning_cfgs['rename_columns'])

        # 4. Map label and subreddit to IDs
        processed['label_desc'] = processed['label'].map(self.labels_map)
        processed['subreddit_id'] = processed['subreddit'].map(self.subreddits_map)

        # 5. Parse timestamps
        if 'timestamp_parsing' in self.cleaning_cfgs:
            timestamp_cfg = self.cleaning_cfgs['timestamp_parsing']
            
            # Extract valid pandas parameters
            pd_params = {}
            if 'dayfirst' in timestamp_cfg:
                pd_params['dayfirst'] = timestamp_cfg['dayfirst']
            if 'primary_format' in timestamp_cfg:
                pd_params['format'] = timestamp_cfg['primary_format']
            if 'error_handling' in timestamp_cfg and timestamp_cfg['error_handling'] == 'coerce':
                pd_params['errors'] = 'coerce'
            
            try:
                processed['timestamp'] = pd.to_datetime(processed['timestamp'], **pd_params)
            except:
                # If primary format fails, try without format specification
                pd_params.pop('format', None)
                processed['timestamp'] = pd.to_datetime(processed['timestamp'], **pd_params)
            # CHECK TYPE
            if not pd.api.types.is_datetime64_any_dtype(processed['timestamp']):
                raise ValueError("Timestamp conversion failed")
            
        # 6. Remove self-replies
        if self.cleaning_cfgs.get('remove_self_replies', False):
            processed = processed[processed['src_author'] != processed['dst_author']]

        # 7. Filter final columns
        final_cols = [
            'subreddit_id', 'subreddit', 'timestamp',
            'submission_id', 'submission_text', 'label', 'label_desc',
            'src_author', 'src_comment_id', 'src_comment_text',
            'dst_author', 'dst_comment_id', 'dst_comment_text',
            'agreement_fraction', 'individual_kappa'
        ]
        return processed[final_cols]
    
    def process_comments(self, replies: pd.DataFrame | None = None) -> pd.DataFrame:
        #=======================================
        # HELPER FUNCTIONS
        #=======================================
        def _extract_unique_comments(replies):
            comments_list = []
            # Source comments (child)
            src_comments = replies[['subreddit_id', 'subreddit', 'timestamp', 'submission_id', 'submission_text',
                              'src_author', 'src_comment_id', 'src_comment_text']].copy()
            src_comments = src_comments.rename(columns={
                'src_author': 'author',
                'src_comment_id': 'comment_id', 
                'src_comment_text': 'comment_text'
            })
            src_comments['is_parent'] = False
            
            # Destination comments (parents)
            dst_comments = replies[['subreddit_id', 'subreddit', 'timestamp', 'submission_id', 'submission_text',
                              'dst_author', 'dst_comment_id', 'dst_comment_text']].copy()
            dst_comments = dst_comments.rename(columns={
                'dst_author': 'author',
                'dst_comment_id': 'comment_id', 
                'dst_comment_text': 'comment_text'
            })
            dst_comments['is_parent'] = True

            # Combine and remove duplicates
            all_comments = pd.concat([src_comments, dst_comments])
            unique_comments = all_comments.drop_duplicates(subset=['comment_id'])
            return unique_comments
        
        def _apply_fixed_windows(days: int, df: pd.DataFrame):
            df = df.copy().sort_values(by='timestamp')
            min_date = df['timestamp'].min()
            max_date = df['timestamp'].max()
            window_edges = pd.date_range(start=min_date, end=max_date + pd.Timedelta(days=days), freq=f'{days}D')
        
            # Add timestep column
            df['timestep'] = pd.cut(
                df['timestamp'],
                bins=window_edges,
                labels=range(len(window_edges)-1),
                right=False
            )
        
            # Create interval labels
            interval_labels = []
            for i in range(len(window_edges)-1):
                start = window_edges[i].date()
                # For last interval, use actual max timestamp
                if i == len(window_edges)-2:
                    end = max_date.date()
                else:
                    end = (window_edges[i+1] - pd.Timedelta(days=1)).date()
                interval_labels.append(f"{start} - {end}")
        
            df['interval'] = df['timestep'].apply(lambda x: interval_labels[int(x)] if pd.notnull(x) else None)
            df['actual_window_size'] = df.groupby(['subreddit_id', 'interval'])['timestamp'].transform(
                lambda ts: (ts.max() - ts.min()).days
            )
        
            return df

        def _apply_custom_subreddit_windows(subreddit_name, df, windows_dict):
            days = windows_dict.get(subreddit_name, 178)
            return _apply_fixed_windows(days, df)

        def _merge_small_windows(df, min_window_size):
            df = df.copy()
            while True:
                window_counts = df['interval'].value_counts().sort_index()
                small_windows = window_counts[window_counts < min_window_size].index.to_list()
                if not small_windows:
                    break
                all_intervals = sorted(df['interval'].dropna().unique())
                merged_any = False
                for sw in small_windows:
                    if sw not in df['interval'].values:
                        continue
                    window_idx = all_intervals.index(sw)
                    # Determine merge direction
                    if window_idx == 0:
                        target = all_intervals[1] if len(all_intervals) > 1 else None
                    elif window_idx == len(all_intervals) - 1:
                        target = all_intervals[-2] if len(all_intervals) > 1 else None
                    else:
                        prev_win = all_intervals[window_idx - 1]
                        next_win = all_intervals[window_idx + 1]
                        prev_count = len(df[df['interval'] == prev_win])
                        next_count = len(df[df['interval'] == next_win])
                        target = prev_win if prev_count <= next_count else next_win
                    if target:
                        target_rows = df[df['interval'] == target]
                        if len(target_rows) == 0:
                            print(f"    + [subreddit: {df['subreddit_id'].iloc[0]}] Skipping merge: target interval '{target}' is empty.")
                            continue
                        # Get all timestamps to update interval label
                        merged_mask = (df['interval'].astype(str) == str(sw)) | (df['interval'].astype(str) == str(target))
                        merged_timestamps = df.loc[merged_mask, 'timestamp']
                        new_start = merged_timestamps.min().date()
                        new_end = merged_timestamps.max().date()
                        new_interval_label = f"{new_start} - {new_end}"
                        target_timestep = target_rows['timestep'].iloc[0]
                        
                        # Print details
                        # n_from_sw = (df['interval'] == sw).sum()
                        # n_from_target = (df['interval'] == target).sum()
                        # old_sw_timestep = df[df['interval'] == sw]['timestep'].unique()
                        # old_target_timestep = df[df['interval'] == target]['timestep'].unique()
                        # print(f"Subreddit: {df['subreddit_id'].iloc[0]}")
                        # print(f"    + Merging {n_from_sw} comments from '{sw}' (timestep:{old_sw_timestep[0]}) "
                        #     f"and {n_from_target} from '{target}' (timestep:{old_target_timestep[0]}) "
                        #     f"onto new interval '{new_interval_label}'"
                        # )
                        
                        # Update both intervals and timestep
                        df.loc[merged_mask, 'interval'] = new_interval_label
                        df.loc[merged_mask, 'timestep'] = target_timestep
                        merged_any = True
                if not merged_any:
                    break
            
            # Recalculate intervals
            intervals_sorted = sorted(df['interval'].dropna().unique())
            interval_map = {interval: i for i, interval in enumerate(intervals_sorted)}
            df['timestep'] = df['interval'].map(interval_map)
            
            # Recalculate actual_window_size after merging
            df['actual_window_size'] = df.groupby('interval')['timestamp'].transform(
                lambda ts: (ts.max() - ts.min()).days
            )

            return df
        
        #=======================================
        # END HELPER FUNCTIONS
        #=======================================
        
        comments = replies.copy()
        # 1. Get unique comments from replies data
        comments = _extract_unique_comments(replies)

        #=======================================
        # PROCESS COMMENTS
        #=======================================
        # 2. Infer parent comment timestamp
        parent_time_cfg = self.temporal_cfgs.get('parent_time_inference', {})
        if parent_time_cfg.get('infer_parent_comment_time', True):
            delta_min = parent_time_cfg.get('delta_minutes', 30)
            if 'is_parent' in comments.columns:
                comments.loc[comments['is_parent'], 'timestamp'] -= pd.Timedelta(minutes=delta_min)
        
        # 3. Apply windowing
        window_cfg = self.temporal_cfgs.get('windowing', {})
        if window_cfg.get('enabled', True):
            window_strategy = window_cfg.get('strategy', None)
            strategy_configs = window_cfg.get(f'{window_strategy}_windows', {})
        
            if window_strategy == 'fixed':
                days = strategy_configs.get('size', 178)
                print(f"Using fixed windows of {days} days")
                comments = comments.groupby('subreddit_id', group_keys=False).apply(
                    lambda df: _apply_fixed_windows(days, df),
                )
            elif window_strategy == 'custom_subreddit':
                print(f"Using custom subreddit windows: {strategy_configs}")
                comments = comments.groupby('subreddit', group_keys=False).apply(
                    lambda df: _apply_custom_subreddit_windows(df.subreddit.iloc[0], df, strategy_configs)
                )

            # 4. Handle merging of small windows
            merge_cfg = window_cfg.get('merge_small_windows', {})
            if merge_cfg.get('enabled', True):
                min_window_size = merge_cfg.get('min_comments_per_window', 50)
                print(f"    + Merging small windows with minimum size: {min_window_size}")
                comments = comments.groupby('subreddit_id', group_keys=False).apply(
                    lambda df: _merge_small_windows(df, min_window_size)
                )

        return comments

    def process_replies(self, replies: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
        replies = replies.copy()
        reply_comments = comments[comments['is_parent'] == False][['subreddit_id', 'comment_id', 'timestep', 'interval', 'actual_window_size']]
        assert len(replies) == len(reply_comments), "Replies and comments must have the same length"

        # 1. Left join replies on subreddit_id and comment_id
        replies_temporal = replies.merge(
            reply_comments,
            left_on=['subreddit_id', 'src_comment_id'],
            right_on=['subreddit_id', 'comment_id'],
            how='left',
        ).drop('comment_id', axis=1)
        
        # 2. Calculate confidence (agreement_fraction * individual_kappa)
        default_confidence = self.default_values['confidence']
        replies_temporal['confidence'] = replies_temporal['agreement_fraction'] * replies_temporal['individual_kappa'].fillna(default_confidence)
        print(f"    + Calculated confidence for {len(replies_temporal)} replies, number of NaNs: {replies_temporal['individual_kappa'].isna().sum()} (used default_confidence = {default_confidence})")

        # 3. Keep minimal final columns (drop _text columns)
        replies_temporal = replies_temporal.loc[:, ~replies_temporal.columns.str.endswith('_text')]

        return replies_temporal

    def process_submissions(self, comments: pd.DataFrame | None = None):
        comments = comments.copy()
        # 1. Get unique submissions from comment data
        submissions = comments.groupby(['subreddit_id', 'subreddit', 'submission_id']).agg({
            'submission_text': 'first',
            'timestamp': 'min',          
            'timestep': 'first',         
            'interval': 'first',         
            'actual_window_size': 'first' 
        }).reset_index()
        submissions = submissions.rename(columns={
            'timestamp': 'first_comment_time',
            'timestep': 'first_comment_timestep',
            'interval': 'first_comment_interval',
            'actual_window_size': 'first_comment_actual_window_size'
        })
        submissions = submissions[['subreddit_id', 'subreddit', 'submission_id', 'submission_text', 
                                'first_comment_time', 'first_comment_timestep', 
                                'first_comment_interval', 'first_comment_actual_window_size']]
        print(f"Total unique submissions: {len(submissions)}")

        return submissions
    
    def process_user_pairs(self, replies: pd.DataFrame) -> pd.DataFrame:
        def _calculate_net_vector(labels, confidences):
            vector = [0.0, 0.0, 0.0]  # [disagree, neutral, agree]
            total_weight = 0.0            
            for label, conf in zip(labels, confidences):
                if label in [0, 1, 2]:
                    vector[label] += conf
                    total_weight += conf
            # Normalize to sum to 1 (if any interactions exist)
            if total_weight > 0:
                vector = [round(x / total_weight, 6) for x in vector]
            return vector

        if replies.empty:
            print("No replies data available to build user pairs.")
            return pd.DataFrame()
        group_cols = ['subreddit_id', 'subreddit', 'timestep', 'interval', 'actual_window_size', 'src_author', 'dst_author']
        agg_dict = {
            'label': list,
            'agreement_fraction': 'mean',
            'individual_kappa': 'mean', 
            'confidence': list,  # Keep all confidence values per group
        }

        # Aggregate user pairs and round values to 3 decimals
        pairs = replies.groupby(group_cols).agg(agg_dict).reset_index().rename(columns={
            'agreement_fraction': 'mean_agreement_fraction',
            'individual_kappa': 'mean_kappa',
        })
        pairs['mean_kappa'] = pairs['mean_kappa'].round(3)
        pairs['mean_agreement_fraction'] = pairs['mean_agreement_fraction'].round(3)

        # Calculate mean confidence and round to 3 decimals
        pairs['mean_confidence'] = pairs['confidence'].apply(lambda x: round(np.mean(x), 3) if x else 0.0)

        # Calculate net vectors using actual confidence values per interaction
        net_vectors = []
        for _, row in pairs.iterrows():
            labels = row['label']
            confidences = row['confidence']
            net_vector = _calculate_net_vector(labels, confidences)
            net_vectors.append(net_vector)
        pairs['net_vector'] = net_vectors

        # Remove intermedia columns
        pairs = pairs.drop(columns=['label', 'confidence'])
        print(f"Built user pairs with {len(pairs)} interactions")
        print(f"   + Unique src_authors: {pairs['src_author'].nunique()}, dst_authors: {pairs['dst_author'].nunique()}")
        print(f"   + Rows with NaN kappa: {pairs['mean_kappa'].isna().sum()}")
        return pairs

    def embed_text_column(self, device, df: pd.DataFrame, text_column: str, configs: dict) -> pd.DataFrame:
        if not configs:
            print("    + No embedding configurations provided, skipping embedding.")
            return df
        texts = df[text_column].tolist()
        if configs.get('type', 'sentence-transformers') == 'sentence-transformers':
            from sentence_transformers import SentenceTransformer
            model_name = configs.get('name', 'all-MiniLM-L6-v2')
            batch_size = configs.get('batch_size', 128)
            max_length = configs['max_length']

            model = SentenceTransformer(model_name)
            if max_length is not None:
                model.max_seq_length = max_length
            print(f"    + Using embedding model: {model_name} max_length: {model.max_seq_length} batch size: {batch_size} on device: {model.device}")

            embeddings = model.encode(
                texts, 
                batch_size=batch_size, 
                show_progress_bar=True, 
                convert_to_numpy=True,
                device=device
            )
            print(f"    + Embedding shape: {embeddings[0].shape}")

            # Add embeddings to df
            df = df.copy()
            df['embeddings'] = list(embeddings)

            # Dump torch cache if used cuda
            if device and str(device) != 'cpu':
                import torch
                torch.cuda.empty_cache()
                print("    + Cleared CUDA cache after embedding.")       
        return df

    def save_processed_data(self, file_type, data, file_path):
        if file_type == 'csv':
            pd.DataFrame.to_csv(data, file_path, index=False)
        elif file_type == 'pickle':
            pd.to_pickle(data, file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}. Use 'csv' or 'pickle'.")
    
    # Print summary functions
    def print_basic_reply_statistics(self, df, group_col='subreddit'):
        stats = []
        for sid, group in df.groupby(group_col):
            earliest = group['timestamp'].min().date()
            latest = group['timestamp'].max().date()
            n_replies = len(group)
            n_self = (group['src_author'] == group['dst_author']).sum()
            pct_self = round(n_self / n_replies, 2) if n_replies else 0.0
            pct_disagree = round((group['label_desc'] == 'disagree').mean() * 100, 1)
            pct_neutral = round((group['label_desc'] == 'neutral').mean() * 100, 1)
            pct_agree = round((group['label_desc'] == 'agree').mean() * 100, 1)
            n_unique_comments = group['src_comment_id'].nunique() + group['dst_comment_id'].nunique()
            n_unique_authors = group['src_author'].nunique() + group['dst_author'].nunique()

            # New timestep summary columns
            total_timesteps = group['timestep'].nunique() if 'timestep' in group.columns else 0
            if 'timestep' in group.columns and 'actual_window_size' in group.columns and total_timesteps > 0:
                avg_timestep_window_size = float(
                    group[['timestep', 'actual_window_size']]
                    .drop_duplicates(subset=['timestep'])['actual_window_size']
                    .mean()
                )
            else:
                avg_timestep_window_size = 0.0

            stats.append({
                group_col: sid,
                'Earliest Date': earliest,
                'Latest Date': latest,
                '# Replies': n_replies,
                '% Self-Replies': pct_self,
                '% Disagree': pct_disagree,
                '% Neutral': pct_neutral,
                '% Agree': pct_agree,
                '# Unique Comments': n_unique_comments,
                '# Unique Authors': n_unique_authors,
                'Total Timesteps': int(total_timesteps),
                'Avg Timestep Window Size (days)': round(avg_timestep_window_size, 1),
            })
        # Add "All" row (use label_desc for consistency)
        all_group = df
        earliest = all_group['timestamp'].min().date()
        latest = all_group['timestamp'].max().date()
        n_replies = len(all_group)
        n_self = (all_group['src_author'] == all_group['dst_author']).sum()
        pct_self = round(n_self / n_replies, 2) if n_replies else 0.0
        pct_disagree = round((all_group['label_desc'] == 'disagree').mean() * 100, 1)
        pct_neutral = round((all_group['label_desc'] == 'neutral').mean() * 100, 1)
        pct_agree = round((all_group['label_desc'] == 'agree').mean() * 100, 1)
        n_unique_comments = all_group['src_comment_id'].nunique() + all_group['dst_comment_id'].nunique()
        n_unique_authors = all_group['src_author'].nunique() + all_group['dst_author'].nunique()

        # Unique windows across all subreddits
        if {'timestep', 'actual_window_size'}.issubset(all_group.columns):
            windows_all = (
                all_group[['subreddit_id', 'timestep', 'actual_window_size']]
                .dropna(subset=['timestep', 'actual_window_size'])
                .drop_duplicates(subset=['subreddit_id', 'timestep'])
            )
            total_timesteps = len(windows_all)
            avg_timestep_window_size = float(windows_all['actual_window_size'].mean()) if total_timesteps > 0 else 0.0
        else:
            total_timesteps = 0
            avg_timestep_window_size = 0.0

        stats.insert(0, {
            group_col: 'All',
            'Earliest Date': earliest,
            'Latest Date': latest,
            '# Replies': n_replies,
            '% Self-Replies': pct_self,
            '% Disagree': pct_disagree,
            '% Neutral': pct_neutral,
            '% Agree': pct_agree,
            '# Unique Comments': n_unique_comments,
            '# Unique Authors': n_unique_authors,
            'Total Timesteps': int(total_timesteps),
            'Avg Timestep Window Size (days)': round(avg_timestep_window_size, 1),
        })
        display(pd.DataFrame(stats))

    def print_subreddit_timestep_info(self, df: pd.DataFrame, count_info: str = "replies"):
        for sid, group in df.groupby('subreddit'):
            print(f"Subreddit: {sid} ({count_info})")
            df_sub = group.groupby(['timestep', 'interval', 'actual_window_size']).size().reset_index(name='total_count')
            min_date = pd.to_datetime(group['timestamp'].min()).date()
            max_date = pd.to_datetime(group['timestamp'].max()).date()
            all_row = {
                'timestep': 'All',
                'interval': f"{min_date} - {max_date}",
                'actual_window_size': (max_date - min_date).days,
                'total_count': len(group)
            }
            df_sub = pd.concat([pd.DataFrame([all_row]), df_sub], ignore_index=True)
            display(df_sub)
    
    # Main processor function
    def run(self, raw_data: pd.DataFrame | None = None, device=None, summarize=True) -> ProcessedData:
        def _summarize():
            print("Basic reply statistics (filtered self-replies):")
            self.print_basic_reply_statistics(replies)
            print("Subreddit timestep info for replies:")
            self.print_subreddit_timestep_info(replies, count_info="replies")
            print("Subreddit timestep info for comments:")
            self.print_subreddit_timestep_info(comments, count_info="comments")
            
        if raw_data is None:
            raw_data = pd.read_csv(self.raw_path)
        else:
            if not isinstance(raw_data, pd.DataFrame):
                raise ValueError("Invalid `raw_data` provided")
                
        if not self.processed_path:
            os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)

        #=========================================================
        # BEGIN PROCESSING
        #=========================================================
        processed_file_names = ['comments.pkl', 'replies.pkl', 'user_pairs.pkl', 'submissions.pkl']
        # If processed files already exists, skip processing and load from cache
        if all(os.path.exists(os.path.join(self.processed_path, fname)) for fname in processed_file_names):
            print(f"Processed files already exist, loading {processed_file_names} from '{self.processed_path}/'")            
            comments = pd.read_pickle(os.path.join(self.processed_path, 'comments.pkl'))
            replies = pd.read_pickle(os.path.join(self.processed_path, 'replies.pkl'))
            user_pairs = pd.read_pickle(os.path.join(self.processed_path, 'user_pairs.pkl'))
            submissions = pd.read_pickle(os.path.join(self.processed_path, 'submissions.pkl'))

            # Summarize
            if summarize:
                _summarize()
            
            return ProcessedData(comments, replies, user_pairs, submissions)

        print("Begin Data Processing...")
        cleaned = self.clean_data(raw_data)
        comments = self.process_comments(cleaned)
        replies = self.process_replies(cleaned, comments)
        user_pairs = self.process_user_pairs(replies)
        submissions = self.process_submissions(comments)

        # Check if configs require text embeddings
        embed_comments = self.embedding_cfgs.get('comments', False)
        embed_submissions = self.embedding_cfgs.get('submissions', False)
        text_emb_cfg = self.embedding_cfgs['model'] if (embed_comments or embed_submissions) else {}
        if embed_comments:
            print(f"Embedding comments...")
            comments = self.embed_text_column(device, comments, 'comment_text', text_emb_cfg)
        if embed_submissions:
            print(f"Embedding submissions...")
            submissions = self.embed_text_column(device, submissions, 'submission_text', text_emb_cfg)

        # Save processed files        
        self.save_processed_data('csv', cleaned, f"{self.processed_path}/deb_label_cleaned.csv")
        print(f"Saved cleaned data to {self.processed_path}/deb_label_cleaned.csv")
        
        self.save_processed_data('pickle', comments, f"{self.processed_path}/comments.pkl")
        print(f"Saved processed comments data to {self.processed_path}/comments.pkl")

        self.save_processed_data('pickle', replies, f"{self.processed_path}/replies.pkl")
        print(f"Saved processed replies data to {self.processed_path}/replies.pkl")
        
        self.save_processed_data('pickle', user_pairs, f"{self.processed_path}/user_pairs.pkl")
        print(f"Saved processed user pairs data to {self.processed_path}/user_pairs.pkl")

        self.save_processed_data('pickle', submissions, f"{self.processed_path}/submissions.pkl")
        print(f"Saved processed submissions data to {self.processed_path}/submissions.pkl")

        if summarize:
            _summarize()
            
        return ProcessedData(comments, replies, user_pairs, submissions)