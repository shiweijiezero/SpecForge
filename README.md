<div align="center" id="sglangtop">
<img src="./assets/logo.png" alt="logo" width="400" margin="10px"></img>

[![documentation](https://img.shields.io/badge/📖-Documentation-red.svg?style=flat)](https://docs.sglang.ai/SpecForge/)
[![github badge](https://img.shields.io/badge/📃%20LMSYS-Blog-black.svg?style=flat)](https://lmsys.org/blog/2025-07-25-spec-forge/)
[![slack badge](https://img.shields.io/badge/Slack-join-blueviolet?logo=slack&amp)](https://sgl-fru7574.slack.com/archives/C09784E3EN6)
[![SGLang Eagle3](https://img.shields.io/badge/🤗%20Hugging%20Face-SGLang%20Eagle3-yellow.svg?style=flat)](https://huggingface.co/collections/lmsys/eagle-3-6886b2329f3998a8bc23f8ed)
[![license](https://img.shields.io/badge/License-MIT%202.0-blue)](./LICENSE)

</div>

## 📍 项目概览

SpecForge 是由 SGLang 团队开发的生态项目，它是一个用于训练投机解码模型的框架，让你能够无缝地将模型移植到 SGLang 推理框架以加速推理。

我们见过许多投机解码的开源项目，但其中多数维护不善或无法直接兼容 SGLang。我们推出这个项目，是希望开源社区能享受到这样一个投机解码框架：
- 由 SpecForge 团队定期维护：代码开箱即用
- 与 SGLang 直接兼容：无需额外移植工作
- 提供高性能训练能力：支持在线/离线/张量并行/FSDP 模式满足多样化需求

查看 [**我们的文档**](https://docs.sglang.ai/SpecForge/) 开始使用。

## 🎉 新闻动态

- [2025-08] 🔔 SpecForge 被列为 LMSYS 的 [旗舰项目](https://lmsys.org/about/)。祝贺 SpecForge 团队！
- [2025-08] 🔥 SpecForge 为 GPT-OSS 提供了 Eagle3 草稿模型。查看 [LMSYS.org](https://lmsys.org/blog/2025-08-27-gpt-oss/) 的博客
- [2025-07] 🔥 SpecForge 与 Llama4-Eagle3 检查点一起发布。查看我们在 [LMSYS.org](https://lmsys.org/blog/2025-07-25-spec-forge/) 的博客

## ✨ 致谢

<img src="./assets/acknowledgements.png" alt="acknowledgements"></img>

我们衷心感谢 EAGLE 官方团队，特别是张洪洋和李宇辉，感谢他们的宝贵贡献和支持。我们也要感谢 NVIDIA 团队，特别是 Avery H 和 Izzy Putterman，以及 Google 团队，特别是王颖，感谢他们在整个项目过程中的深刻讨论和慷慨帮助。

我们特别感谢美团的大力支持和有意义的贡献，这在推动项目前进中发挥了至关重要的作用。

本项目也受到了 LLM 社区许多优秀开源项目的启发，包括 [EAGLE](https://github.com/SafeAILab/EAGLE)、[BaldEagle](https://github.com/NickL77/BaldEagle) 和 [TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) 等。他们的贡献和共享的知识极大地促进了我们的工作。

## 💡 特别感谢 Voltage Park

我们衷心感谢 [Voltage Park](https://www.voltagepark.com/)，我们的官方基础设施合作伙伴。作为与 SGLang 团队正式合作的一部分，Voltage Park 提供了关键的 GPU 资源，使我们能够高效可靠地训练和评估大规模投机解码模型。这一合作关系对 SpecForge 的实现至关重要。我们深深感谢 Voltage Park 致力于使尖端 AI 基础设施更易获得的使命，并期待在我们推动开源 LLM 服务和优化边界的过程中继续合作。

## 📃 引用

```bibtex
@misc{specforge2025,
  title={SpecForge: Train speculative decoding models effortlessly},
  author={Shenggui Li, Yikai Zhu, Chao Wang, Fan Yin, Shuai Shi, Yubo Wang, Yi Zhang, Yingyi Huang, Haoshuai Zheng, Yineng Zhang},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/sgl-project/specforge}},
}
