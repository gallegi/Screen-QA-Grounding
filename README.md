# VLMs on ScreenQA Dataset

This project focuses on building and evaluating Vision Language Models (VLMs) for visual question answering with grounding using the ScreenQA dataset. The models are trained to understand and answer questions about screen interfaces while providing visual grounding for their responses.

## Project Overview

This repository contains code and resources for training and evaluating Vision Language Models on the ScreenQA dataset, specifically focusing on the task of answering questions with visual grounding. The models learn to:
- Process screen interface images
- Understand natural language questions
- Generate accurate answers
- Provide visual grounding/localization for their responses

## Dataset

The ScreenQA dataset consists of:
- Screen interface images from various mobile applications
- Natural language questions about the interface elements
- Ground truth answers
- Bounding box annotations for visual grounding

## Features

- Implementation of state-of-the-art VLM architectures
- Training pipeline for visual question answering
- Evaluation metrics including:
  - Answer accuracy
  - Visual grounding precision
  - Response relevance
- Visualization tools for model predictions and attention maps

## Installation
### Download ScreenQA dataset and code
```bash
git clone https://github.com/google-research-datasets/screen_qa.git
cd screen_qa
mkdir RICO
cd RICO
wget https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/unique_uis.tar.gz
tar -xvf unique_uis.tar.gz
cd ../..
```
### Install requirements
```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
```

### Evaluation

```bash
```

## Model Architecture

The project implements/uses the following VLM architectures:
- Architecture 1
- Architecture 2
- Custom modifications for screen understanding

## Results

| Model | Answer Accuracy | Grounding Precision | Response Time |
|-------|----------------|---------------------|---------------|
| Model1 | XX% | XX% | XXms |
| Model2 | XX% | XX% | XXms |

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Submit a Pull Request

## Citation

If you use this code or the models in your research, please cite:

```bibtex
@article{screenqa2024,
  title={Visual Grounding on Screen QA with VLMs},
  author={The Nam Nguyen},
  journal={Conference/Journal Name},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ScreenQA dataset creators
- Research community contributions
- Supporting institutions and grants

## Contact

For questions or issues, please:
- Open an issue in this repository
- Contact: namnguyen61031@gmail.com
