# Craw: A Unified and Efficient Querying Framework for Large-Scale Video Datasets

This repository contains the code for the paper:

**Craw: A Unified and Efficient Querying Framework for Large-Scale Video Datasets**

## Environment Setup

### Requirements

- Python 3.11.13 (recommended)
- NVIDIA GPU with CUDA support (≥8GB VRAM)
- RAM ≥16GB

### Installation

Clone this repository:

```bash
git clone https://github.com/CrawZhou/Craw_Submission.git
cd Craw_Submission/sys
```

Create and activate a conda environment:

```bash
conda create -n your_name python=3.11.13 (recommended)
conda activate your_name
```

Install dependencies:

```bash
pip install -r requirements.txt
```



## Data Preparation

### Experimental Datasets

Craw was evaluated on multiple video datasets to demonstrate its scalability and performance. The following datasets were used in our experiments:

1. [Highway](https://www.youtube.com/watch?v=KBsqQez-O4w)
2. [Resort](https://favyen.com/miris/)
3. [Bellevue](https://github.com/City-of-Bellevue/TrafficVideoDataset)
4. [Warsaw](https://favyen.com/miris/)

### Custom Videos

You can use your own videos in `.mp4`, `.avi`, or `.mkv` format. Organize as:

```
/path/to/videos/
├── video1.mp4
├── video2.mp4
└── ...
```

---

## Execution

### Video Processing

The `--full` flag runs the complete pipeline: object tracking, segmentation → MSU → indexing, clustering. Results are stored in the specified `--db` directory. Below are some examples:

Run the full pipeline on a video file:

```bash
# Process video and store results in ./msu_db
python -m run --video "path/to/video.mp4" --full

# Specify database path
python -m run --video "path/to/video.mp4" --db "./my_db" --full

# From existing frame_info JSON (skip tracking)
python -m run --frame-info "path/to/frame_info.json" --db "./msu_db" --full

# With ROI (x1,y1,x2,y2)
python -m run --video "path/to/video.mp4" --roi "100,100,600,400" --full

# With custom YOLO model
python -m run --video "path/to/video.mp4" --model-path "path/to/model.pt" --full

```

more options can be found in `run.py`

### Query Examples

Query examples use an existing database (created with `--full`) to search and retrieve similar MSUs.

```bash
# Query similar MSUs by ID
python -m run --db "./msu_db" --query --msu-id 0 --top-k 5

# Query similar video segments
python -m run --db "./msu_db" --query-video "path/to/query_video.mp4" --top-k 5

# Search by object class in F matrix (e.g., class 0 = pedestrian)
python -m run --db "./msu_db" --element-type F --element-value 0

# Search by interaction type in I matrix (e.g., type 2 = follow)
python -m run --db "./msu_db" --element-type I --element-value 2
```


## Baseline Usage

Please refer to the official tutorials of the following baselines:

1. [MIRIS](https://github.com/favyen/miris)
2. [FIGO](https://github.com/jiashenC/FiGO)
3. [EQUI-VOCAL](https://github.com/uwdb/equi-vocal)
4. [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)

For Video-zilla, the implementation is included in the `sys` folder and integrates with Craw’s pipeline.


## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
