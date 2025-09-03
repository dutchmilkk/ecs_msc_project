# DebateGNN: A Multitask Graph Neural Network for Echo Chamber Detection in Online Debates

**Author:** Viriya Duch Sam (240761532)  
**Institution:** MSc Artificial Intelligence, School of Electronic Engineering and Computer Science, Queen Mary University of London  
**Supervisor:** Prof. Iran Roman

## Abstract

Echo chambers (ECs) are often measured through endorsement proxies such as retweets or likes, which capture homophily but potentially overlook argumentative dynamics. This thesis examines whether incorporating stance information into message passing alters how ECs are quantified in debate settings, using Reddit reply graphs where stance signals (agree, neutral, disagree) shape community structure. We introduce DebateGNN, a hybrid Graph Neural Network that combines GraphSAGE and Edge-Conditioned Convolutions. Edges encode annotator-derived stance vectors and confidence scores, and the model is trained with a multitask objective for link, confidence and stance predictions. Using the DEBAGREEMENT dataset of stance-labelled Reddit replies, we compute the Echo Chamber Score (ECS) over temporal graph snapshots and compare DebateGNN embeddings to those from the GCN-based EchoGAE baseline. Results show that EchoGAE generally produces higher ECS values and closer alignment with structural partitions, while DebateGNN yields typically lower trajectories and performs better in low-modularity, disagreement-heavy settings. We conclude that ECS outcomes depend on modelling choices, and recommend reporting both topology-led and stance-aware estimates.

## Repository Structure

```
├── src/
│   ├── models/
│   │   └── multitask_debate_gnn.py    # Main DebateGNN implementation
│   ├── baselines/
│   │   └── echogae/                   # EchoGAE baseline
│   ├── analysis/
│   ├── modules/
│   ├── utils/
│   └── visualization/
├── data/
│   ├── processed/                     # Processed graph data
│   └── raw/                          # Original DEBAGREEMENT dataset
├── configs/                          # Configuration files
├── checkpoints/                      # Trained model weights
├── results/                          # Experimental results
├── train_gnn.ipynb                   # Model training notebook
├── temporal_ecs_analysis.ipynb       # Echo chamber analysis
└── ablation_*.ipynb                  # Ablation studies
```

## Key Features

- **Multitask Architecture**: Joint learning of link prediction, confidence estimation and stance prediction
- **Edge-Conditioned Convolutions**: Incorporates stance and confidence information into message passing
- **Temporal Analysis**: Processes sequential graph snapshots for echo chamber evolution
- **Echo Chamber Quantification**: Implements Echo Chamber Score (ECS) computation
- **Comparative Analysis**: Benchmarks against EchoGAE baseline (Alatawi et al., 2024)

## Quick Start

### Environment Setup

**Option 1: Conda (Recommended)**
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate ecs_project
```

**Option 2: Pip**
```bash
# Install dependencies
pip install -r requirements.txt
```

### Running the Code

1. Train DebateGNN: Run `train_gnn.ipynb`
2. Analyze echo chambers: Run `temporal_ecs_analysis.ipynb`
3. View ablation studies: Run `ablation_*.ipynb` notebooks

## Model Architecture

The `MultitaskDebateGNN` combines:
- GraphSAGE layers for neighborhood aggregation
- Edge-Conditioned Convolutions (`ECCConv`) for stance-aware message passing
- Separate prediction heads for link, confidence, and stance tasks
- Uncertainty quantification and multitask loss balancing

## Citation

```bibtex
@mastersthesis{duchsam2025debategnn,
  title={DebateGNN: A Multitask Graph Neural Network for Echo Chamber Detection in Online Debates},
  author={Duch Sam, Viriya},
  year={2025},
  school={Queen Mary University of London},
  type={MSc Thesis}
}
```
