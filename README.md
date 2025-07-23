![](./Illustrations/Banner.png)

### TL;DR

If you only have limited time to learn Artificial Intelligence, here’s what I recommend:

- 📘 **Read this book**: [AI Engineering: Building Applications with Foundation Models](https://www.oreilly.com/library/view/ai-engineering/9781098166298/)  
- 🎥 **Watch this video**: [Deep Dive into LLMs like ChatGPT](https://youtu.be/7xTGNNLPyMI?si=aUTq_qUzyUx36BsT)  
- 🧠 **Follow this course**: [🤗 Agents Course](https://huggingface.co/learn/agents-course/)

If you want more (and there’s a lot more) keep reading.

## Why this repo exists
Learning often feels like walking down a road that forks every few meters; you’re always exploring, never really arriving. And that’s the beauty of it.

When I was working in games, people would ask me: “How do I learn to make games?” My answer was always: “Pick a game, and build it, learn the tools and concepts along the way.” I’ve taken the same approach with AI.

This repository is a collection of the material I’ve used (and continue to use) to learn AI: books, courses, papers, tools, models, datasets, and notes. It’s not a curriculum, it’s more like a journal. One that’s helped me build, get stuck, and keep going.

Do I know AI? Not really. But I’m learning, building, and having a great time doing it.

I hope something in here is useful to you too. And if you have suggestions or feedback, I’d love to hear it.

## Books
<img src="./Illustrations/Characters/NI_Cool.png" alt="NI Cool" align="right" width="200px">

Here are the books I've read to make sense of AI/ML/DL. Some are more technical, some are more theoretical, and some are more ideas.

They are not in any particular order, although I tried to group them together based on what i think makes sense.

- [Hands-On Large Language Models: Language Understanding and Generation](https://www.oreilly.com/library/view/hands-on-large-language/9781098150952/) Building an LLM from scratch is difficult, even understanding existing open-source options can be challenging. This book does a great job of explaining how LLMs work and introduces common architectures at a deep enough level to be practical without being overwhelming.
- [AI Engineering: Building Applications with Foundation Models](https://www.oreilly.com/library/view/ai-engineering/9781098166298/) If you feel lost and don’t know where to start, this book can serve as a great map. Chip explains the most common concepts behind AI in a clear and approachable way.
- [Deep Learning - A Visual Approach](https://www.glassner.com/portfolio/deep-learning-a-visual-approach/) Probably the best resource out there for building solid intuition about the many concepts surrounding deep learning. Andrew, the author, did a wonderful job illustrating these concepts, making it much easier to develop a real understanding of them.
- [Practical Deep Learning: A Python-Based Introduction](https://nostarch.com/practical-deep-learning-python) Probably the best resource for balancing deep learning concepts with a hands-on, Python-based approach. It’s much easier to follow if you actively implement the code, even if that just means typing out the examples from the book.
- [Hands-On Generative AI with Transformers and Diffusion Models](https://www.oreilly.com/library/view/hands-on-generative-ai/9781098149239/) This book provides a panoramic view of different model types to generate images, audio, and text. It shows a lot of code, and comes with challenges that are interesting to tackle if you want to get hands-on experience. Omar(one of the authors) is an ex-Hugging Face, now working at Google Deep Mind. 
- [Dive Into Data Science: Use Python To Tackle Your Toughest Business Challenges](https://nostarch.com/dive-data-science) Not strictly about AI, although if you think about it, AI is deeply tied to data science. This book is great for understanding real-world scenarios and how to approach them using AI tools and Python. For me, the data preparation part was especially helpful.
- [Dive into Deep Learning](https://d2l.ai/index.html) An amazing free book that covers the fundamental concepts of deep learning, including how to build your own models from scratch. It leans toward the heavier side, but it comes with plenty of Jupyter notebooks you can run to test your understanding. I still can’t believe this resource is free.
- [Grokking Deep Learning](https://www.manning.com/books/grokking-deep-learning) When I read this book, I had a lot of “aha” moments. I’m not sure if it was the clarity of the explanations or the fact that I already had some background from other books, but either way, the examples and explanations are excellent. It strikes a good balance between theory, algorithms, and practical implementation.
- [Prompt Engineering for LLMs](https://www.oreilly.com/library/view/prompt-engineering-for/9781098156145/) This book provides useful tips and tricks to program using LLMs, essentially via prompts. As of 2025, I think the techniques discussed here, can be very useful for people using LLMs as part of their programming stack.
- [Math for Deep Learning: A Practitioner's Guide to Mastering Neural Networks](https://nostarch.com/math-deep-learning) If you’re interested in how AI works under the hood, it ultimately comes down to a lot of math. This book does a great job explaining the essential mathematics behind implementing neural networks. It can feel overwhelming at times. The book is incredibly helpful for understanding how popular frameworks work, and how to better evaluate them for your use cases.
- [Why Machines Learn: The Elegant Math Behind Modern AI](https://anilananthaswamy.com/why-machines-learn) A fascinating mix of math and history that explores how we arrived at large language models (LLMs) today (2024). I recommend reading it after you’ve built some basic math foundations — you’ll find it much more helpful that way. 
- [Understanding Deep Learning](https://udlbook.github.io/udlbook/) Another amazing book, and it’s completely free! (But buy it if you can). This book puts special focus on the pedagogy of how to learn (and teach) deep learning. It also comes with interactive notebooks so you can try out the explanations yourself.
- [Practical Deep Learning for Coders with Fastai and PyTorch: AI Applications Without a PhD](https://course.fast.ai/Resources/book.html) This book comes with video lectures by [Jeremy Howard](https://x.com/jeremyphoward), the author and creator of FastAI. The goal is to give coders a head start in deep learning without diving too deep into the weeds. It still covers how to achieve a lot using the FastAI framework and PyTorch. Having the videos included made it especially cool, since I could both listen to and watch the author explain the concepts.
- [The Shape of Data: Geometry-Based Machine Learning and Data Analysis in R](https://nostarch.com/shapeofdata) When I got to this book, I already had a decent understanding of machine learning. Reading its geometric approach(spatial, geometrical, and visual rather than only statistical) gave me many moments where I smiled, things made sense even when framed through a different paradigm. I do wish the code examples weren’t in R, but I still managed to follow the concepts easily.
- [The Art of Machine Learning: A Hands-On Guide to Machine Learning with R](https://nostarch.com/art-machine-learning) I would recommend this book to anyone who is already familiar with R or wants to learn both R and machine learning together. It’s a quick read.
- [How AI Works: From Sorcery to Science](https://nostarch.com/how-ai-works) I recommend this book if you don’t want to code or dive into the technical details of AI, deep learning, or machine learning, but still want a high-level understanding of what it’s all about. It’s great for people in non-engineering or non-technical roles. The book gives clear explanations of how things work without getting lost in the weeds.
- [Superintelligence: Paths, Dangers, Strategies](https://global.oup.com/academic/product/superintelligence-9780198739838) This was the first non-technical book I read on AI. It was fascinating because it offered a glimpse into the broader implications of what we’re building. I don’t think everything predicted in the book will happen, but it presents great perspectives on how to mitigate potential risks.
- [The Myth of Artificial Intelligence](https://www.hup.harvard.edu/books/9780674278660) This book offers the perspective that artificial intelligence won’t become the powerful, dangerous new form of life many imagine. It provides solid arguments for why that is, presenting a view that is much less apocalyptic than what you usually hear.
- [The Coming Wave: Technology, Power, and the Twenty-first Century's Greatest Dilemma](https://the-coming-wave.com/) This book explores more catastrophic scenarios. Scenarios that, in my opinion, are entirely feasible. The book warns how the AI race could go wrong. It highlights not just the risks from machines themselves, but also the dangers of powerful tools falling into the wrong hands.
- [Everything Is Predictable: How Bayesian Statistics Explain Our World](https://www.goodreads.com/en/book/show/199798096-everything-is-predictable) This book provides a compelling view at how everything in life can be predicted, which essentially is what we are making machines do. Predict data.
- [Ways of Being: Animals, Plants, Machines: The Search for a Planetary Intelligence](https://draw-down.com/products/animals-plants-machines-the-search-for-a-planetary-intelligence) This book explores not just artificial intelligence, but the many forms of intelligence that already exist alongside us on Earth. It’s a great source of inspiration and a reminder that intelligence comes in many shapes, most of which we don't understand.

## Courses & Tutorials
<img src="./Illustrations/Characters/AI_Teacher.png" alt="AI Teacher" align="right" width="200px">

Not everyone learns the same way, sometimes I get too tired of just reading, and tutorials or courses in video form make me feel like I’m talking to someone.
If you prefer learning through videos or more interactive formats, I recommend taking a look at the following materials:

- [Getting Started with Deep Learning](https://learn.nvidia.com/courses/course?course_id=course-v1:DLI+S-FX-01+V1) A short introduction to deep learning, delivered by NVIDIA. If you just want a quick glimpse of the very basics before jumping into higher-level implementation, this is a solid place to start.
- [MIT Intro to Deep Learning](http://introtodeeplearning.com/) A free, intensive bootcamp taught by MIT researchers. Getting direct access to this content (updated every time they teach it) is amazing. (~10 hours of deep learning concepts, plus interesting guest lectures.)
- [Practical Deep Learning for Coders](https://course.fast.ai/) This is the accompanying course version of the FastAI book by [Jeremy Howard](https://x.com/jeremyphoward).
- [C++ Neural Network in a Weekend](https://www.jeremyong.com/cpp/machine-learning/2020/10/23/cpp-neural-network-in-a-weekend/) This might be too much if you’re just starting out, or if you’re not interested in low-level C++ implementations of neural networks. But I found it fascinating, and it made me appreciate how far modern frameworks and APIs have come.
- [🤗 Agents Course](https://huggingface.co/learn/agents-course/) Now that agentic AI is trending, Hugging Face launched this free course showcasing their `smolagents` framework. It also covers LlamaIndex and LangGraph. 


## Videos & Talks
I find it increasingly difficult these days to stay focused on videos. Maybe it’s because watching usually means being on a computer or phone, and distractions are always just a click away.  
However, the videos listed here are so well-made, well-researched, and genuinely interesting that I believe they’re not just useful, but sticky.
One more thing I appreciate: in the world of AI, many of the best video talks and tutorials often come directly from the people actually building the models, frameworks, and tools. What a great time to be learning.

- [Andrej Karpathy: Software Is Changing (Again)](https://youtu.be/LCEmiRjPEtQ?si=3osbmpa4XjRDazJa) This session is great session outlining how the way we talk to computers is changing. We're still just transforming data, but the way we do so, is changing. I think this is particularly useful for software developers, Computer Scientists, and students who wonder if a CS career is worth it in 2025.
- [Transformers (how LLMs work) explained visually](https://youtu.be/wjZofJX0v4M?si=AdcRXv8GtgIZTqkw) 3Blue1Brown’s beautifully and clearly explained video on how large language models (LLMs) work, including the transformer architecture and the concept of embeddings.
- [Visualizing transformers and attention | Talk for TNG Big Tech Day '24](https://youtu.be/KJtZARuO3JY?si=rWDYNLAteyWnhHju) Grant Sanderson's (3Blue1Brown) live explanation on how to think visually about transformers. This session is at the intersection of art and science.
- [Deep Dive into LLMs like ChatGPT](https://youtu.be/7xTGNNLPyMI?si=aUTq_qUzyUx36BsT) Aderej Karpathy's(OpenAI, Tesla, Stanford) masterclass on how LLMs work. In-depth, but super accesible.
- [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2095s) Andrej Karpathy's step-by-step guide on building GPT.

## Tools, Frameworks & Platforms
<img src="./Illustrations/Characters/NI_Architect.png" alt="AI Architect" align="right" width="200px">

Understanding all the tools, frameworks, architectures, and ecosystems around AI can sometimes feel harder than understanding AI itself. Below are the ones I’ve explored and used enough to feel confident recommending.  
Of course, these won’t solve every use case, and I’m not listing every supporting technology you might need to build real-world AI systems, but it’s a start.

| Category         | Tools  |
|------------------|--------|
| Core Frameworks   |   - [**Hugging Face**](https://huggingface.co/): A hub, that hosts models, datasets, apps, and communities around AI. <br> - [**Ollama**](https://ollama.com/): Run LLMs locally in your computer(CLI). <br> - [**LM Studio**](https://lmstudio.ai/): Discover, and run LLMs in your computer, using a UI.|
| Developer Tools   |   - [**Gradio**](https://www.gradio.app/): Create ML-powered apps for the web. Easy to use UI API. <br> - [**Google Colab**](https://colab.research.google.com/): You have seen probably many resources use Jupyter Notebooks, this platform allows you to run them. <br> - [**MongoDB**](https://mongodb.com/): Database that allows you to perform vector search. |

## Python Libraries & Toolkits
<img src="./Illustrations/Characters/AI_Aha.png" alt="AI Aha" align="right" width="200px">

AI goes far beyond any single language, operating system, hardware, or framework. There are countless implementations across different programming languages, runtimes, and platforms. From my experience, though, Python is what most people use and teach.  
Following that path, I’ve focused most of my learning around Python as well. That said, similar libraries (and often the same ones) likely exist for your favorite environment too.

| Category                     | Libraries                  |
|-------------------------------|-----------------------------|
| Data Science & Computation    | Pandas, NumPy, SciPy, scikit-learn |
| Plotting & Visualization      | Matplotlib, Seaborn         |
| Machine Learning / Deep Learning | TensorFlow, PyTorch       |
| Image Processing              | Pillow                      |
| Web Scraping                  | Beautiful Soup, Selenium    |

## Models
<img src="./Illustrations/Characters/NI_Wee.png" alt="NI Wee" align="right" width="200px">

At the core of deep learning are the models. Some are general-purpose large language models (LLMs), while others are specialized for specific tasks like text generation, image creation, or coding.

These are the models I’ve used or explored:
- [GPT (OpenAI)](https://platform.openai.com/docs/models)
- [Claude (Anthropic)](https://www.anthropic.com/api)
- [Gemini (Google DeepMind)](https://deepmind.google/technologies/gemini/)
- [LLaMA (Meta)](https://www.llama.com/)
- [DeepSeek](https://huggingface.co/deepseek-ai)
- [Meta Segment Anything Model 2](https://ai.meta.com/sam2/)

If you're looking to explore beyond these, I recommend checking out the following model hubs. They host a wide variety of models with different licenses and for many use cases:
- [Hugging Face Models Hub](https://huggingface.co/models)
- [TensorFlow Hub](https://www.tensorflow.org/hub)
- [NVIDA NGC Catalog](https://catalog.ngc.nvidia.com/models)
- [Roboflow Universe](https://universe.roboflow.com/)

## Articles, Blogs & Interviews

- [AI Is Nothing Like a Brain, and That’s OK](https://www.quantamagazine.org/ai-is-nothing-like-a-brain-and-thats-ok-20250430) - 04/30/25
- [John Carmack: Different path to AGI](https://dallasinnovates.com/exclusive-qa-john-carmacks-different-path-to-artificial-general-intelligence/) - 02/02/23
- Deep Learning in a Nutshell [Part 1: Core Concepts](https://developer.nvidia.com/blog/deep-learning-nutshell-core-concepts/) | [Part 2:  History and Training](https://developer.nvidia.com/blog/deep-learning-nutshell-history-training/) | [Part 3: Sequence Learning](https://developer.nvidia.com/blog/deep-learning-nutshell-sequence-learning/) | [Part 4: Reinforcement Learning](https://developer.nvidia.com/blog/deep-learning-nutshell-reinforcement-learning/) | [Part 5: Reinforcement Learning](https://developer.nvidia.com/blog/deep-learning-nutshell-reinforcement-learning/) - 11/03/2015

## Papers
<img src="./Illustrations/Characters/AI_Scientist.png" alt="AI Scientist" align="right" width="200px">

At the core of most AI advances is research — deep, complex work published by people pushing the boundaries of what’s possible. These breakthroughs often appear in the form of academic papers.
Reading papers can be overwhelming at first. It’s not always easy to unpack their meaning or follow the math. I suggest using tools like [NotebookLM](https://notebooklm.google.com/) or joining a local AI paper-reading club.

- [Agentic Reasoning and Tool Integration for LLMs via Reinforcement Learning](https://www.arxiv.org/abs/2505.01441)
- [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
- [PrimitiveAnything](https://huggingface.co/spaces/hyz317/PrimitiveAnything)
- [DreamFusion: Text-to-3D using 2D Diffusion](https://dreamfusion3d.github.io/)
- [GET3D: A Generative Model of High Quality 3D Textured Shapes Learned from Images](https://nv-tlabs.github.io/GET3D/)
- [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)
- [Text2Room: Extracting Textured 3D Meshes from 2D Text-to-Image Models](https://lukashoel.github.io/text-to-room/)
- [Imagen: Text-to-Image Diffusion Models](https://imagen.research.google/)
- [Gen-1](https://arxiv.org/abs/2302.03011)
- [Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification](https://proceedings.mlr.press/v81/buolamwini18a/buolamwini18a.pdf)
- [Uncovering and Mitigating Algorithmic Bias through Learned Latent Structure](http://introtodeeplearning.com/AAAI_MitigatingAlgorithmicBias.pdf)

## Datasets
<img src="./Illustrations/Characters/AI_Question.png" alt="AI Question" align="right" width="200px">

The current approach to teaching machines relies heavily on data — often, massive amounts of it. In some cases, we use datasets that were created and labeled by humans. In others, we rely on synthetic data generated by machines, or a combination of both.
This section includes some well-known datasets you can explore and use to train your models. Platforms like Hugging Face also host a wide range of datasets for different tasks and domains.

| Name                                                                                                          | Domain                                      |
|---------------------------------------------------------------------------------------------------------------|---------------------------------------------|
| [Kaggle Datasets](https://www.kaggle.com/datasets/)                                                           | Various / General ML                        |
| [CelebA](https://paperswithcode.com/dataset/celeba)                                                           | Computer Vision / Facial Attributes         |
| [COCO](https://cocodataset.org/#home)                                                                         | Computer Vision / Object Detection          |
| [ImageNet](https://image-net.org/)                                                                            | Computer Vision / Classification            |
| [Cityscapes Dataset](https://www.cityscapes-dataset.com/)                                                     | Computer Vision / Segmentation              |
| [ObjectNet](https://objectnet.dev/)                                                                           | Computer Vision / Robustness Testing        |
| [LAION 5B](https://laion.ai/blog/laion-5b/)                                                                   | Multimodal / Vision-Language                |
| [NAIRR Datasets](https://nairrpilot.org/pilotresources)                                                       | Various / Research Datasets                 |
| [UCI Machine Learning Datasets](https://archive.ics.uci.edu/datasets)                                         | Traditional ML / Tabular                    |
| [Common Crawl](https://commoncrawl.org/)                                                                      | NLP / Web-Scale Corpus                      |
| [The Pile](https://pile.eleuther.ai/)                                                                         | NLP / Language Modeling                     |
| [C4 (Colossal Clean Crawled Corpus)](https://www.tensorflow.org/datasets/community_catalog/huggingface/c4)    | NLP / Pretraining Corpus                    |

## Notes & Highlights
- [Standard Notations for Deep Learning](https://cs230.stanford.edu/files/Notation.pdf)
- [AI Index Report | Stanford](https://hai.stanford.edu/research/ai-index-report)
- [Historical data on 'notable' Models by Epoch](https://epoch.ai/data/notable-ai-models)
- [Ethics of AI](https://ethics-of-ai.mooc.fi/)

<div align="center">
  <img src="./Illustrations/Characters/AI_Sleeping.png" alt="AI Sleeping" width="200px">
</div>
