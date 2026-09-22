# WearableDeviceAgents

An LLM-powered, task-oriented multi-agent framework for complex wearable health analysis. It answers natural-language questions about structured wearable records through querying, analysis, and health-support advice.

## Framework

<p align="center">
  <img src="assets/framework.png" width="900">
</p>

## Query Agent

<p align="center">
  <img src="assets/query_agent.png" width="850">
</p>

## Installation

Use Python 3.12 (the repository specifies 3.12.13), preferably in a virtual environment.

```bash
git clone https://github.com/yangkunpeng-coder/WearableDeviceAgents.git
cd WearableDeviceAgents
python -m pip install -r requirements.txt
```

The dependency file uses the Tencent Cloud PyPI mirror and includes an unconditional `pywin32` dependency, which may prevent installation on non-Windows systems.

## Configuration

Set API keys in your process environment before starting Python:

| Environment variable | Used by |
| --- | --- |
| `DEEPSEEK_API_KEY` | Required for the examples below. `brain_s/deepseek_v4_pro.py` calls `deepseek-v4-pro` at `https://api.deepseek.com`. |
| `DASHSCOPE_API_KEY` | Optional; used when selecting `model_type='qwen'` through `agent_s/agent_config.py`. |

The AgentScope configuration in `agent_s/agent_config.py` uses `deepseek-chat` or `qwen3-max`; it is separate from the examples' DeepSeek brain. There is no `.env.example` or automatic `.env` loading in the application code.

## Paper

**A Task-Oriented Multi-Agent Framework for Complex Wearable Health Analysis**

Author: Kunpeng Yang

arXiv: [https://arxiv.org/abs/2609.24107](https://arxiv.org/abs/2609.24107)

DOI: [https://doi.org/10.48550/arXiv.2609.24107](https://doi.org/10.48550/arXiv.2609.24107)

Category: Multiagent Systems (cs.MA)

## Citation

```bibtex
@article{yang2026taskoriented,
  title={A Task-Oriented Multi-Agent Framework for Complex Wearable Health Analysis},
  author={Yang, Kunpeng},
  journal={arXiv preprint arXiv:2609.24107},
  year={2026},
  doi={10.48550/arXiv.2609.24107}
}
```

## Disclaimer

This repository is a research prototype.

It is not a medical device and should not be used for diagnosis, treatment, or professional medical decision-making.
