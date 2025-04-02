> This repo fored from [SparkAudio/Spark-TTS](https://github.com/SparkAudio/Spark-TTS/).


**添加的功能:**

- 改进命令行交互，方便使用。并将声音特征提取和音频生成拆分开可独立使用。

- **声音特征保存**：从参考音频中提取说话人全局特征tokens，保存至本地文件。后续生成时可直接加载保存的特征tokens应用于TTS，无需重新从音频中提取。

- **文本拆分处理**：将输入的长文本，使用分段算法将文本按句子结束标点、暂停标点和字符/单词数量限制拆分，逐段生成音频后合并。利用缓存机制支持中断后继续生成。

- **音频段缓存**：使用哈希算法基于文本内容和TTS参数（模型、语音特征、性别、音调、语速等）生成缓存键，存储已生成的音频段。再次处理相同文本和参数时直接从缓存读取，避免重复推理。

- <del>**语音参数调节**：提供gender（male/female）、pitch（very_low到very_high）、speed（very_low到very_high）和emotion参数设置，通过模型prompt控制输出音频特性。</del> （无效）

**命令行调用方式**

`python . --help`


---

## Spark-TTS 🔥

### Overview

Spark-TTS is an advanced text-to-speech system that uses the power of large language models (LLM) for highly accurate and natural-sounding voice synthesis. It is designed to be efficient, flexible, and powerful for both research and production use.

### Key Features

- **Simplicity and Efficiency**: Built entirely on Qwen2.5, Spark-TTS eliminates the need for additional generation models like flow matching. Instead of relying on separate models to generate acoustic features, it directly reconstructs audio from the code predicted by the LLM. This approach streamlines the process, improving efficiency and reducing complexity.
- **High-Quality Voice Cloning**: Supports zero-shot voice cloning, which means it can replicate a speaker's voice even without specific training data for that voice. This is ideal for cross-lingual and code-switching scenarios, allowing for seamless transitions between languages and voices without requiring separate training for each one.
- **Bilingual Support**: Supports both Chinese and English, and is capable of zero-shot voice cloning for cross-lingual and code-switching scenarios, enabling the model to synthesize speech in multiple languages with high naturalness and accuracy.
- **Controllable Speech Generation**: Supports creating virtual speakers by adjusting parameters such as gender, pitch, and speaking rate.

---


## Install
**Clone and Install**

  Here are instructions for installing on Linux. If you're on Windows, please refer to the [Windows Installation Guide](https://github.com/SparkAudio/Spark-TTS/issues/5).  
*(Thanks to [@AcTePuKc](https://github.com/AcTePuKc) for the detailed Windows instructions!)*


- Clone the repo
``` sh
git clone https://github.com/SparkAudio/Spark-TTS.git
cd Spark-TTS
```

- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html
- Create Conda env:

``` sh
conda create -n sparktts -y python=3.12
conda activate sparktts
pip install -r requirements.txt
# If you are in mainland China, you can set the mirror as follows:
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```

**Model Download**

Download via python:
```python
from huggingface_hub import snapshot_download

snapshot_download("SparkAudio/Spark-TTS-0.5B", local_dir="pretrained_models/Spark-TTS-0.5B")
```

Download via git clone:
```sh
mkdir -p pretrained_models

# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

git clone https://huggingface.co/SparkAudio/Spark-TTS-0.5B pretrained_models/Spark-TTS-0.5B
```

**Basic Usage**

You can simply run the demo with the following commands:
``` sh
cd example
bash infer.sh
```

Alternatively, you can directly execute the following command in the command line to perform inference：

``` sh
python -m cli.inference \
    --text "text to synthesis." \
    --device 0 \
    --save_dir "path/to/save/audio" \
    --model_dir pretrained_models/Spark-TTS-0.5B \
    --prompt_text "transcript of the prompt audio" \
    --prompt_speech_path "path/to/prompt_audio"
```

## Runtime

**Nvidia Triton Inference Serving**

We now provide a reference for deploying Spark-TTS with Nvidia Triton and TensorRT-LLM. The table below presents benchmark results on a single L20 GPU, using 26 different prompt_audio/target_text pairs (totalling 169 seconds of audio):

| Model | Note   | Concurrency | Avg Latency     | RTF | 
|-------|-----------|-----------------------|---------|--|
| Spark-TTS-0.5B | [Code Commit](https://github.com/SparkAudio/Spark-TTS/tree/4d769ff782a868524f29e0be851ca64f8b22ebf1/runtime/triton_trtllm) | 1                   | 876.24 ms | 0.1362|
| Spark-TTS-0.5B | [Code Commit](https://github.com/SparkAudio/Spark-TTS/tree/4d769ff782a868524f29e0be851ca64f8b22ebf1/runtime/triton_trtllm) | 2                   | 920.97 ms | 0.0737|
| Spark-TTS-0.5B | [Code Commit](https://github.com/SparkAudio/Spark-TTS/tree/4d769ff782a868524f29e0be851ca64f8b22ebf1/runtime/triton_trtllm) | 4                   | 1611.51 ms | 0.0704|


Please see the detailed instructions in [runtime/triton_trtllm/README.md](runtime/triton_trtllm/README.md ) for more information.


## **Demos**

Here are some demos generated by Spark-TTS using zero-shot voice cloning. For more demos, visit our [demo page](https://sparkaudio.github.io/spark-tts/).

---

<table>
<tr>
<td align="center">
    
**Donald Trump**
</td>
<td align="center">
    
**Zhongli (Genshin Impact)**
</td>
</tr>

<tr>
<td align="center">

[Donald Trump](https://github.com/user-attachments/assets/fb225780-d9fe-44b2-9b2e-54390cb3d8fd)

</td>
<td align="center">
    
[Zhongli](https://github.com/user-attachments/assets/80eeb9c7-0443-4758-a1ce-55ac59e64bd6)

</td>
</tr>
</table>

---

<table>

<tr>
<td align="center">
    
**陈鲁豫 Chen Luyu**
</td>
<td align="center">
    
**杨澜 Yang Lan**
</td>
</tr>

<tr>
<td align="center">
    
[陈鲁豫Chen_Luyu.webm](https://github.com/user-attachments/assets/5c6585ae-830d-47b1-992d-ee3691f48cf4)
</td>
<td align="center">
    
[Yang_Lan.webm](https://github.com/user-attachments/assets/2fb3d00c-abc3-410e-932f-46ba204fb1d7)
</td>
</tr>
</table>

---


<table>
<tr>
<td align="center">
    
**余承东 Richard Yu**
</td>
<td align="center">
    
**马云 Jack Ma**
</td>
</tr>

<tr>
<td align="center">

[Yu_Chengdong.webm](https://github.com/user-attachments/assets/78feca02-84bb-4d3a-a770-0cfd02f1a8da)

</td>
<td align="center">
    
[Ma_Yun.webm](https://github.com/user-attachments/assets/2d54e2eb-cec4-4c2f-8c84-8fe587da321b)

</td>
</tr>
</table>

---


<table>
<tr>
<td align="center">
    
**刘德华 Andy Lau**
</td>
<td align="center">

**徐志胜 Xu Zhisheng**
</td>
</tr>

<tr>
<td align="center">

[Liu_Dehua.webm](https://github.com/user-attachments/assets/195b5e97-1fee-4955-b954-6d10fa04f1d7)

</td>
<td align="center">
    
[Xu_Zhisheng.webm](https://github.com/user-attachments/assets/dd812af9-76bd-4e26-9988-9cdb9ccbb87b)

</td>
</tr>
</table>


---

<table>
<tr>
<td align="center">
    
**哪吒 Nezha**
</td>
<td align="center">
    
**李靖 Li Jing**
</td>
</tr>

<tr>
<td align="center">

[Ne_Zha.webm](https://github.com/user-attachments/assets/8c608037-a17a-46d4-8588-4db34b49ed1d)
</td>
<td align="center">

[Li_Jing.webm](https://github.com/user-attachments/assets/aa8ba091-097c-4156-b4e3-6445da5ea101)

</td>
</tr>
</table>


## To-Do List

- [x] Release the Spark-TTS paper.
- [ ] Release the training code.
- [ ] Release the training dataset, VoxBox.


## Citation

```
@misc{wang2025sparktts,
      title={Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens}, 
      author={Xinsheng Wang and Mingqi Jiang and Ziyang Ma and Ziyu Zhang and Songxiang Liu and Linqin Li and Zheng Liang and Qixi Zheng and Rui Wang and Xiaoqin Feng and Weizhen Bian and Zhen Ye and Sitong Cheng and Ruibin Yuan and Zhixian Zhao and Xinfa Zhu and Jiahao Pan and Liumeng Xue and Pengcheng Zhu and Yunlin Chen and Zhifei Li and Xie Chen and Lei Xie and Yike Guo and Wei Xue},
      year={2025},
      eprint={2503.01710},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2503.01710}, 
}
```


## ⚠️ Usage Disclaimer

This project provides a zero-shot voice cloning TTS model intended for academic research, educational purposes, and legitimate applications, such as personalized speech synthesis, assistive technologies, and linguistic research.

Please note:

- Do not use this model for unauthorized voice cloning, impersonation, fraud, scams, deepfakes, or any illegal activities.

- Ensure compliance with local laws and regulations when using this model and uphold ethical standards.

- The developers assume no liability for any misuse of this model.

We advocate for the responsible development and use of AI and encourage the community to uphold safety and ethical principles in AI research and applications. If you have any concerns regarding ethics or misuse, please contact us.