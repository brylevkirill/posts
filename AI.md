  https://habr.com/en/post/436458

---


  The biggest issue with AI is not that it is stupid but lack of definition for intelligence and hence lack of measure for it [[1a]](https://arxiv.org/abs/0712.3329) [[1b]](https://amazon.com/Measure-All-Minds-Evaluating-Intelligence/dp/1107153018).

  Turing test is not a good measure because gorilla Koko wouldn't pass though she could solve more problems than many disabled human beings [[2a]](https://youtube.com/user/kokoflix/videos) [[2b]](https://blog.piekniewski.info/2017/04/13/ai-confuses-intelligent).

  It is quite possible that people in the future might wonder why people back in 2019 thought that an agent trained to play a fixed game in a simulated environment such as Go had any intelligence [[3a]](https://blog.piekniewski.info/2017/04/13/ai-confuses-intelligent) [[3b]](http://blog.piekniewski.info/2016/08/09/intelligence-is-real) [[3c]](https://blog.piekniewski.info/2016/10/10/elephant-in-the-chinese-room) [[3d]](https://blog.piekniewski.info/2016/11/15/ai-and-the-ludic-fallacy) [[3e]](http://blog.piekniewski.info/2017/01/13/outside-the-box) [[3f]](http://hunch.net/?p=9828091) [[3g]](http://hunch.net/?p=8825714) [[3h]](https://slideslive.com/38922025/deep-reinforcement-learning-1?t=4250).

  Intelligence is more about applying/transferring old knowledge to new tasks (playing Quake Arena good enough without any training after mastering Doom) than compressing agent's experience into heuristics to predict game outcome and determining agent's action to maximize outcome value in given game state (playing Quake Arena good enough after million games after mastering Doom) [[4]](https://youtu.be/DSYzHPW26Ig?t=6m28s).

  Human intelligence is about ability to adapt to the physical/social world, and playing Go is a particular adaptation performed by human intelligence, and developing algorithm to learn to play Go is more performant adaptation, and developing mathematical theory to play Go might be even more performant. <cut/>

  It makes more sense to compare a human and AI not by effectiveness and efficiency of end product of adaptation (in games played between human and agent) but by effectiveness and efficiency of process of adaptation (in games played between human-coded agent and machine-learned agent after limited practice) [[5]](https://arxiv.org/abs/1911.01547).

  Dota 2, StarCraft 2, Civilization 5 and probably even GTA 5 might be solved in not so distant future but ability to play any new game at human level with no prior training would be way more significant.

  The second biggest issue with AI is lack of robustness in a long tail of unprecedented situations (including critical ones in healthcare, self-driving vehicles, finance) which at present can't be handled with accuracy even close to acceptable [[6a]](http://blog.piekniewski.info/2017/01/13/outside-the-box) [[6b]](http://blog.piekniewski.info/2017/07/27/measuring-performance-ml) [[6c]](http://blog.piekniewski.info/2017/03/06/give-me-a-dataset-to-train-and-i-shall-move-the-world) [[6d]](https://blog.piekniewski.info/2017/04/03/the-complexity-of-simplicity) [[6e]](https://youtu.be/uyZOcUDhIbY?t=17m27s).

  Complex models exploit any patterns that relate input to output variables but some patterns might not hold for cases poorly covered by training data [section "progress"] [[7a]](https://blog.piekniewski.info/2019/04/07/deep-learning-and-shallow-data) [[7b]](http://blog.piekniewski.info/2017/03/06/give-me-a-dataset-to-train-and-i-shall-move-the-world) [[7c]](https://blog.keras.io/the-limitations-of-deep-learning.html) [[7d]](https://thebestschools.org/magazine/limits-of-modern-ai). >99% of healthcare applications use simple models like logistic regression (domain knowledge is converted into code to compute statistics as features) so as to avoid spurious correlations and gain more robustness on outliers [[8a]](https://www.sciencedirect.com/science/article/pii/S0895435618310813) [[8b]](https://twitter.com/david_sontag/status/1009629318635491330).

  For an agent in simulated environment like Go or Quake, true model of environment is either known or available so that agent can generate any amount of training data in order to learn how to act optimally in any situation. Finding out correlations in that data isn't intelligent — for real-world problems discovering true model is key [[9a]](http://blog.piekniewski.info/2016/11/03/reactive-vs-predictive-ai) [[9b]](https://blog.piekniewski.info/2016/11/01/statistics-and-dynamics) [[9c]](https://blog.keras.io/the-limitations-of-deep-learning.html) [[9d]](http://emanuelderman.com/the-limitations-of-machine-learning-in-finding-the-laws-of-nature) [[9e]](https://facebook.com/icml.imls/videos/2265408103721327?t=1094).

  For an organism, the real world is not a fixed game with known environment and rules such as Go or Quake but a game with environment and rules largely unknown and always changing [[10]](https://blog.piekniewski.info/2017/04/13/ai-confuses-intelligent). It has to adapt to unexpected changes of environment and rules including changes caused by adversaries. It has to be capable of wide autonomy as opposed to merely automation necessary to play some fixed game.

  It might turn out to be impossible to have self-driving vehicles and humanoid robots operating alongside humans without training them to obtain human-level adaptability to the real world. It might turn out to be impossible to have personal assistants substituting humans in key aspects of their lives without training them to obtain human-level adaptability to the social world [[11a]](http://rodneybrooks.com/forai-steps-toward-super-intelligence-ii-beyond-the-turing-test) [[11b]](http://rodneybrooks.com/forai-steps-toward-super-intelligence-iii-hard-things-today) [[11c]](http://rodneybrooks.com/forai-steps-toward-super-intelligence-iv-things-to-work-on-now).




### knowledge vs intelligence

  Knowledge is some information, such as data from observations or experiments, compressed and represented in some computable form, such as text in natural language, mathematical theory in semi-formal language, program in formal language, weights of artificial neural network or synapses of brain.

  Knowledge is about tools (theory, program, physical process) to solve problems. Intelligence is about applying (transferring) and creating (learning) knowledge [[12]](https://youtu.be/6-Uiq8-wKrg?t=3m56s). There is knowledge how to solve problem (a program for computers, a manual for humans), and then there is process of applying knowledge (executing program by computers, inferring, interpreting and executing instructions by humans), and then there is process of creating knowledge (inductive inference/learning from observations and experiments, deductive reasoning from inferred theories and learned models - either by computers or humans).

  Alpha(Go)Zero is way closer to knowledge how to solve particular class of problems than to an intelligent agent capable of applying and creating knowledge. It is a search algorithm like IBM Deep Blue with heuristics being not hardcoded but being tuned during game sessions. It can't apply learned knowledge to other problems - even playing on smaller Go board. It can't create abstract knowledge useful to humans - even simple insight on Go tactic.

  TD-Gammon from 1992 is considered by many as the biggest breakthrough in AI [[13a]](https://youtu.be/N8_gVrIPLQM?t=7m32s) [[13b]](https://youtu.be/ld28AU7DDB4?t=42m7s). TD-Gammon used TD(λ) algorithm with online on-policy updates. TD-Gammon's author used its variation to learn IBM Watson's wagering strategy [[13c]](https://youtube.com/watch?v=7rIf2Njye5k). Alpha(Go)Zero is also roughly a variation of TD(λ) [[13d]](http://techtalks.tv/talks/on-td-learning-and-links-with-deeprl-in-atari-and-alphago/63031). TD-Gammon used neural network trained by Temporal Difference learning with target values calculated using tree search with depth not more than three and using outcomes of games played to the end as estimates of leaf values. Alpha(Go)Zero used deep neural network trained by Temporal Difference learning with target values calculated using Monte-Carlo Tree Search with much bigger depth and using estimates of leaf values and policy actions calculated by network without playing games to the end.

  Qualitative differences between Backgammon and Go as problems and between TD-Gammon and Alpha(Go)Zero as solutions (scale of neural network and number of played games being major differences) are not nearly as big as qualitative differences between perfect information games such as Go and imperfect information games such as Poker (Alpha(Go)Zero being not applicable to Poker and DeepStack to Go/Chess).

  IBM Watson, the most advanced question answering system by far in 2011, is not an intelligent agent. It is knowledge represented as thousands lines of manually coded logic for searching and manipulating sequences of words as well as generating hypotheses and gathering evidences, plus few hundred parameters tuned with linear regression for weighing in different pieces of knowledge for each supported type of question and answer [[14a]](https://youtube.com/watch?v=DywO4zksfXw) [[14b]](https://youtube.com/watch?v=3G2H3DZ8rNc) [[14c]](https://www.youtube.com/watch?v=1n-cwezu8j4). It's not that much different conceptually from database engines which use statistics of data and hardcoded threshold values to construct a plan for executing given query via selecting and pipelining a subset of implemented algorithms for manipulating data.

  IBM Watson can apply its logic for textual information extraction and integration (internal knowledge) to new texts (external knowledge). However, it can't apply its knowledge to problems other than limited factoid question answering without being coded to do so by humans. It can be coded to search for evidences in support of hypotheses in papers on cancer but only using human coded logic to interpret texts (extracting and matching relevant words) and never going beyond that to interpret texts on its own (learning model of the world and mapping texts to simulations on that model). The first approach to interpret texts was sufficient for Jeopardy! [[15]](https://youtu.be/Whtt2H5_isM?t=1h4m31s) but it is not nearly enough when there is no single simple answer. There is huge difference between making conclusions using statistical properties of texts and using statistical properties of real world phenomena estimated with simulations on learned model of that phenomena.

  IBM Watson can't create new knowledge - it can deduce simple facts from knowledge sources (texts and knowledge bases) using human-coded algorithms but it can't induce a theory from the sources and check its truth. WatsonPaths hypothesizes a causal graph using search for texts relevant to case [[16a]](https://youtu.be/1n-cwezu8j4?t=54m42s) [[16b]](https://aaai.org/ojs/index.php/aimagazine/article/view/2715) but inference chaining as approach to reasoning can't be sufficiently robust - inferences have to be checked with simulations or experiments as done by brain.




### what is intelligence?

  Biologists define intelligence as ability to find non-standard solutions for non-standard problems (to deal with unknown unknowns) and distinguish it from reflexes and instincts defined as standard solutions for standard problems [[17a]](http://blog.piekniewski.info/2017/01/13/outside-the-box) [[17b]](http://blog.piekniewski.info/2016/11/15/ai-and-the-ludic-fallacy). Playing Go can't be considered a non-standard problem for AlphaGo after playing millions of games. Detecting new malware can be considered a non-standard problem with no human-level solution so far.

  Most AI researchers focus on top-down approach to intelligence with end-to-end training, i.e. defining objective for high-level problem (such as maximizing expected probability of win by Alpha(Go)Zero) and expecting agent to learn good solutions for low-level subproblems [[18]](http://blog.piekniewski.info/2016/11/03/reactive-vs-predictive-ai). This approach works for relatively simple problems like fixed games in simulated environments but requires an enormous amount of training episodes (several orders of magnitude more than amount which can be experienced by agent in the real world) and leads to solutions incapable of generalization (AlphaGo model trained on 19x19 board is effectively useless for 9x9 board without full retraining). Hardest high-level problems which can be solved by humans are open-ended - humans don't search in fixed space of possible solutions unlike AlphaGo [[19]](https://blog.piekniewski.info/2016/11/15/ai-and-the-ludic-fallacy). Being informed and guided by observations and experiments in the real world, humans come up with good subproblems, e.g. special and general relativity.

  Some AI researchers [section "possible directions"] focus on bottom-up approach, i.e. starting with low-level objectives (such as maximizing ability to predict environment dynamics, including effect of agent's actions on environment), then adding higher-level objectives for agent's intrinsic motivation (such as maximizing learning progress or maximizing available options), and only then adding high-level objectives for problems of interest (such as maximizing game score). This approach is expected to lead to more generalizable and robust solutions for high-level problems because learning with such low-level objectives might lead an agent to also learn self-directing and self-correcting behavior helpful in non-standard or dangerous situations with zero information about them effectively provided by high-level objective. Necessity to adapt/survive provides optimization objectives for organisms to guide self-organization and learning/evolution [[20a]](https://nature.com/articles/s41467-019-11786-6) [[20b]](https://nature.com/articles/s42256-019-0103-7), and some organisms can set up high-level objectives for themselves after being trained/evolved to satisfy low-level objectives. It is quite possible that some set of universal low-level objectives might be derived from a few equations governing flow of energy and information [[21a]](https://arxiv.org/abs/1906.01563), so that optimization with those objectives [section "possible directions"] might lead to intelligence of computers in an analogous way to how evolution of the Universe governed by laws of physics leads to intelligence of organisms [[21b]](https://toddhylton.net/2017/10/of-men-and-machines.html).

  While solving high-level problems in simulated environments such as Go had successes, solving low-level problems such as vision and robotics are yet to have such successes. Humans can't learn to play Go without first learning to discern board and to place stones. Computers can solve some high-level problems without ability to solve low-level ones when high-level problems are abstracted away from low-level subproblems by humans [[22a]](https://blog.piekniewski.info/2016/10/10/elephant-in-the-chinese-room). It is low-level problems which are more computationally complex for both humans and computers although not necessarily more complex as mathematical or engineering problems [[22b]](https://en.wikipedia.org/wiki/Moravec%27s_paradox). It is low-level problems which are prerequisites for commonsense reasoning, i.e. estimating plausibility of arbitrary hypothesis from obtained or imagined observations and from all previously acquired knowledge, which is necessary for machine to adapt to arbitrary environment and to solve arbitrary high-level problem in that environment [[22c]](https://blog.piekniewski.info/2016/08/09/intelligence-is-real) [[22d]](https://blog.piekniewski.info/2017/04/13/ai-confuses-intelligent).




### obstacles

  The first biggest obstacle to applications in real-world environments as opposed to simulated ones seems to be underconstrained objectives for optimization, since the real world is a way more complex environment than any video game due to amount of moving parts and interactions as well as amount of adapting agents. Any sufficiently complex model trained with insufficiently constrained objective will exploit any pattern found in training data that relates input to target variables but spurious correlations won't necessarily generalize to testing data [section "progress"] [[23a]](https://blog.piekniewski.info/2019/04/07/deep-learning-and-shallow-data) [[23b]](http://blog.piekniewski.info/2017/03/06/give-me-a-dataset-to-train-and-i-shall-move-the-world) [[23c]](https://blog.keras.io/the-limitations-of-deep-learning.html) [[24d]](https://thebestschools.org/magazine/limits-of-modern-ai). Even billion examples don't constrain optimization sufficiently and don't lead to major performance gains in image recognition [[24a]](https://arxiv.org/abs/1707.02968) [[24b]](https://arxiv.org/abs/1805.00932). Agents find surprising ways to exploit simulated environments to maximize objectives not constrained enough to prevent exploits [[25a]](https://youtu.be/jAPJeJK18mw?t=10m28s) [[25b]](https://youtube.com/watch?v=-p7VhdTXA0k).

  One way to constrain optimization sufficiently in order to avoid non-generalizable and non-robust solutions is more informative data for training, for example, using physics of the real world or dynamics of the social world as sources of signal as opposed to simulated environments with artificial agents — not nearly as complex and not representative of corner cases in the real/social world. Another way is more complex objective for optimization, for example, learning to predict not only statistics of interest, such as future cumulative rewards conditionally on agent's next actions, but also dynamics, i.e. some arbitrary future properties of environment conditionally on some arbitrary hypothetical future events including agent's next actions [[26a]](http://blog.piekniewski.info/2016/11/01/statistics-and-dynamics) [[26b]](https://blog.piekniewski.info/2016/11/03/reactive-vs-predictive-ai) [[26c]](https://blog.piekniewski.info/2016/11/30/learning-physics-is-the-way-to-go) [[26d]](https://youtu.be/DSYzHPW26Ig?t=6m28s). States and rewards correspond to agent's statistical summaries for interactions with environment while dynamics corresponds to agent's knowledge about how environment works [[27a]](https://youtu.be/6-Uiq8-wKrg?t=3m56s) [[27b]](https://youtu.be/fztxE3Ga8kU?t=58m5s). Progress in learning to predict dynamics of environment [section "possible directions"] [[28a]](https://youtu.be/h7F5sCLIbKQ?t=8m1s) [[28b]](https://youtu.be/ElyFDUab30A?t=17m37s) [[28c]](https://deepmind.com/blog/article/unsupervised-learning) as well as progress in creating options to influence environment [section "possible directions"] [[28d]](https://youtube.com/watch?v=PL0Xq0FFQZ4) [[28e]](https://youtube.com/watch?v=dIUq3LYLEQo) [[28f]](https://blog.piekniewski.info/2016/08/09/intelligence-is-real) might be some of the most powerful kinds of intrinsic motivation for agent and are another ways to constrain optimization.

  The second biggest obstacle seems to be an enormous gap between complexity of simulated environments available for present computers and complexity of real-world environments available for present robots so that agent trained in simulated environment can't be transferred to robot in real-world environment with acceptable performance and robustness [[29]](https://youtube.com/watch?v=7A_QPGcjrh0). Boston Dynamics team never used machine learning to control their robots - they use real-time solvers of differential equations to calculate dynamics and optimal control for models of robots and environments which are not learned from data but specified manually [[30]](https://quora.com/What-kind-of-learning-algorithms-are-used-on-Boston-Dynamics-robots/answer/Eric-Jang). MIT researchers didn't use machine learning to control their robot in DARPA Robotics Challenge 2015, and their robot was the only robot which didn't fall or need physical assistance from humans [[31a]](https://youtube.com/watch?v=2GW7ozcUCFE). A tail event might be not learnable by statistical model [[31b]](https://amazon.com/Black-Swan-Improbable-Robustness-Fragility/dp/081297381X), i.e. through forming a separating hyperplane of neural network as a decision boundary for action, and might require non-statistical inference, i.e. through inducing a model/theory for the event and drawing/checking hypotheses using it. Thus not only statistics but dynamics of phenomena might have to be calculated - model might have to be programmed or trained to simulate dynamics of phenomena [[31c]](https://youtu.be/0Tj6CeYCmzw?t=9m20s).

  It's quite possible that the only way to train/evolve agents with intelligence sufficient for hard problems in the real world (such as robotics) and in the social world (such as natural language understanding) might turn out to be:  
  (1) to train/evolve agents in environments which provide as much constraints for optimization as the real and social world (i.e. agents have to be robots operating in the real world alongside humans);  
  (2) to train/evolve agents on problems which provide as much constraints for optimization as the hardest problems solved by organisms in the real world (i.e. agents have to learn to survive as robots in the real world without any direct assistance from humans) and solved by humans in the social world (i.e. agents have to learn to reach goals in the real world using communication with humans as the only tool).




### progress

  Arguably during Deep Learning renaissance period there hasn't been progress in real-world problems such as robotics and language understanding nearly as significant as in fixed games running in simulated environments.

  Opinions on progress of AI research from some of the most realistic researchers:  

  *Michael I. Jordan*  [[32a]](https://medium.com/@mijordan3/artificial-intelligence-the-revolution-hasnt-happened-yet-5e1d5812e1e7) [[32b]](https://youtu.be/4inIBmY8dQI?t=13m9s) [[32c]](https://youtu.be/Tkl6ERLWAbA?t=3m17s)  
  *Rodney Brooks*  [[33a]](https://rodneybrooks.com/forai-steps-toward-super-intelligence-i-how-we-got-here) [[33b]](http://rodneybrooks.com/the-seven-deadly-sins-of-predicting-the-future-of-ai)  
  *Philip Piekniewski*  [[34a]](http://blog.piekniewski.info/2016/08/02/80p_there) [[34b]](https://blog.piekniewski.info/2018/06/20/rebooting-ai-postulates)  
  *Francois Chollet*  [[35a]](https://blog.keras.io/the-limitations-of-deep-learning.html) [[35b]](https://medium.com/@francois.chollet/the-impossibility-of-intelligence-explosion-5be4a9eda6ec)  
  *John Langford*  [[36a]](http://hunch.net/?p=9828091) [[36b]](http://hunch.net/?p=9604328)  
  *Alex Irpan*  [[37]](https://www.alexirpan.com/2018/02/14/rl-hard.html)  


  Deep Learning methods are very non-robust in image understanding tasks [papers on generalization and adversarial examples below] [[38a]](https://objectnet.dev/index.html) [[38b]](https://thegradient.pub/the-limitations-of-visual-deep-learning-and-how-we-might-fix-them) [[38c]](https://blog.piekniewski.info/2016/08/12/how-close-are-we-to-vision) [[38d]](https://blog.piekniewski.info/2016/08/18/adversarial-red-flag) [[38e]](https://blog.piekniewski.info/2016/12/29/can-a-deep-net-see-a-cat) [[38f]](https://petewarden.com/2018/07/06/what-image-classifiers-can-do-about-unknown-objects).  
  Deep Learning methods haven't come even close to replacing radiologists [[39a]](https://twitter.com/lpachter/status/999772075622453249) [[39b]](https://twitter.com/zacharylipton/status/999395902996516865) [[39c]](https://medium.com/@jrzech/what-are-radiological-deep-learning-models-actually-learning-f97a546c5b98).  
  Deep Learning methods are very non-robust in text understanding tasks [[40a]](https://thegradient.pub/frontiers-of-generalization-in-natural-language-processing) [[40b]](https://motherboard.vice.com/en_us/article/j5npeg/why-is-google-translate-spitting-out-sinister-religious-prophecies) [[40c]](https://arxiv.org/abs/1902.01007) [[40d]](https://arxiv.org/abs/1907.07355).  
  Deep Learning methods can't pass first levels of the hardest Atari game [[41a]](https://youtube.com/watch?v=_zbg9rs5QZY) [[41b]](https://medium.com/@awjuliani/on-solving-montezumas-revenge-2146d83f0bc3).  

  ["ObjectNet: A Large-Scale Bias-controlled Dataset for Pushing the Limits of Object Recognition Models"](https://papers.nips.cc/paper/9142-objectnet-a-large-scale-bias-controlled-dataset-for-pushing-the-limits-of-object-recognition-models)  
  ["Approximating CNNs with Bag-of-local-Features Models Works Surprisingly Well on ImageNet"](https://openreview.net/forum?id=SkfMWhAqYQ)  
  ["Measuring the Tendency of CNNs to Learn Surface Statistical Regularities"](https://arxiv.org/abs/1711.11561)  
  ["Excessive Invariance Causes Adversarial Vulnerability"](https://arxiv.org/abs/1811.00401)  
  ["Do Deep Generative Models Know What They Don't Know?"](https://arxiv.org/abs/1810.09136)  
  ["Do ImageNet Classifiers Generalize to ImageNet?"](http://people.csail.mit.edu/ludwigs/papers/imagenet.pdf)  
  ["Do CIFAR-10 Classifiers Generalize to CIFAR-10?"](https://arxiv.org/abs/1806.00451)  
  ["Deep Learning for Segmentation of Brain Tumors: Impact of Cross‐institutional Training and Testing"](https://aapm.onlinelibrary.wiley.com/doi/abs/10.1002/mp.12752)  
  ["Confounding Variables Can Degrade Generalization Performance of Radiological Deep Learning Models"](https://arxiv.org/abs/1807.00431)  
  ["Natural Adversarial Examples"](https://arxiv.org/abs/1907.07174)  
  ["One Pixel Attack for Fooling Deep Neural Networks"](https://arxiv.org/abs/1710.08864)  
  ["A Rotation and a Translation Suffice: Fooling CNNs with Simple Transformations"](https://arxiv.org/abs/1712.02779)  
  ["Semantic Adversarial Examples"](https://arxiv.org/abs/1804.00499)  
  ["Why Do Deep Convolutional Networks Generalize so Poorly to Small Image Transformations?"](https://arxiv.org/abs/1805.12177)  
  ["The Elephant in the Room"](https://arxiv.org/abs/1808.03305)  
  ["Strike (with) a Pose: Neural Networks Are Easily Fooled by Strange Poses of Familiar Objects"](https://arxiv.org/abs/1811.11553)  
  ["Universal Adversarial Triggers for Attacking and Analyzing NLP"](https://arxiv.org/abs/1908.07125)  
  ["Semantically Equivalent Adversarial Rules for Debugging NLP Models"](https://acl2018.org/paper/1406)  
  ["Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference"](https://arxiv.org/abs/1902.01007)  
  ["Probing Neural Network Comprehension of Natural Language Arguments"](https://arxiv.org/abs/1907.07355)  




### possible directions

  *Juergen Schmidhuber*

> "Data becomes temporarily interesting by itself to some self-improving, but computationally limited, subjective observer once he learns to predict or compress the data in a better way, thus making it subjectively simpler and more beautiful. Curiosity is the desire to create or discover more non-random, non-arbitrary, regular data that is novel and surprising not in the traditional sense of Boltzmann and Shannon but in the sense that it allows for compression progress because its regularity was not yet known. This drive maximizes interestingness, the first derivative of subjective beauty or compressibility, that is, the steepness of the learning curve. It motivates exploring infants, pure mathematicians, composers, artists, dancers, comedians, yourself, and artificial systems."

  Intelligence can be viewed as compression efficacy: the more one can compress data, the more one can understand it. Example of increase in compression efficacy: 1. raw observations of planetary orbits 2. geocentric Ptolemaic epicycles 3. heliocentric ellipses 4. Newtonian mechanics 5. general relativity 6.? Under this view, compression of data is understanding, improvement of compressor is learning, progress of improvement is intrinsic reward. To learn as fast as possible about a piece of data, one should decrease as rapidly as possible the number of bits one need to compress that data. If one can choose which data to observe or create, one should interact with environment in a way to obtain data that maximizes the decrease in bits — the compression progress — of everything already known.

  ["The Simple Algorithmic Principle behind Creativity, Art, Science, Music, Humor"](https://youtube.com/watch?v=h7F5sCLIbKQ&t=7m12s)  
  ["Formal Theory of Fun and Creativity"](http://videolectures.net/ecmlpkdd2010_schmidhuber_ftf)  

  ["Formal Theory of Creativity and Fun and Intrinsic Motivation"](http://people.idsia.ch/~juergen/creativity.html)  
  ["Active Exploration, Artificial Curiosity & What's Interesting"](http://people.idsia.ch/~juergen/interest.html)  

  ["Driven by Compression Progress: A Simple Principle Explains Essential Aspects of Subjective Beauty, Novelty, Surprise, Interestingness, Attention, Curiosity, Creativity, Art, Science, Music, Jokes"](http://arxiv.org/abs/0812.4360)  
  ["Formal Theory of Creativity, Fun, and Intrinsic Motivation"](http://people.idsia.ch/~juergen/ieeecreative.pdf)  
  ["Curiosity Driven Reinforcement Learning for Motion Planning on Humanoids"](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3881010/pdf/fnbot-07-00025.pdf)  
  ["What's Interesting?"](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.45.6362)  
  ["PowerPlay: Training an Increasingly General Problem Solver by Continually Searching for the Simplest Still Unsolvable Problem"](http://arxiv.org/abs/1112.5309)  
  ["Unsupervised Minimax: Adversarial Curiosity, Generative Adversarial Networks, and Predictability Minimization"](https://arxiv.org/abs/1906.04493)  


  *Alex Wissner-Gross*

> "Intelligent system needs to optimize future causal entropy, or to put it in plain language, maximize the available future choices. Which in turn means minimizing all the unpleasant situations with very few choices. This makes sense from evolutionary point of view as it is consistent with the ability to survive, it is consistent with what we see among humans (collecting wealth and hedging on multiple outcomes of unpredictable things) and generates reasonable behavior in several simple game situations."

  ["An Equation for Intelligence"](https://youtube.com/watch?v=PL0Xq0FFQZ4)  
  ["The Physics of Artificial General Intelligence"](https://youtube.com/watch?v=dIUq3LYLEQo)  

  ["Intelligence is Real"](http://blog.piekniewski.info/2016/08/09/intelligence-is-real)  
  ["Intelligence Confuses the Intelligent"](https://blog.piekniewski.info/2017/04/13/ai-confuses-intelligent)  

  ["Causal Entropic Forces"](http://math.mit.edu/~freer/papers/PhysRevLett_110-168702.pdf)


  *Filip Piekniewski*

> "By solving a more general problem of physical prediction (to distinguish it from statistical prediction), the input and label get completely balanced and the problem of human selection disappears altogether. The label in such case is just a time shifted version of the raw input signal. More data means more signal, means better approximation of the actual data manifold. And since that manifold originated in the physical reality (no, it has not been sampled from a set of independent and identically distributed gaussians), it is no wonder that using physics as the training paradigm may help to unravel it correctly. Moreover, adding parameters should be balanced out by adding more constraints (more training signal). That way, we should be able to build a very complex system with billions of parameters (memories) yet operating on a very simple and powerful principle. The complexity of the real signal and wealth of high dimensional training data may prevent it from ever finding "cheap", spurious solutions. But the cost we have to pay, is that we will need to solve a more general and complex task, which may not easily and directly translate to anything of practical importance, not instantly at least."

  ["Predictive Vision Model - A Different Way of Doing Deep Learning"](https://youtube.com/watch?v=rBtlQM_pPi0)

  ["Rebooting AI - Postulates"](http://blog.piekniewski.info/2018/06/20/rebooting-ai-postulates)  
  ["Intelligence Confuses The Intelligent"](http://blog.piekniewski.info/2017/04/13/ai-confuses-intelligent)  
  ["Intelligence Is Real"](http://blog.piekniewski.info/2016/08/09/intelligence-is-real)  
  ["AI And The Ludic Fallacy"](http://blog.piekniewski.info/2016/11/15/ai-and-the-ludic-fallacy)  
  ["The Peculiar Perception Of The Problem Of Perception"](http://blog.piekniewski.info/2016/08/23/the-peculiar-perception-of-the-problem-of-perception)  
  ["Statistics And Dynamics"](http://blog.piekniewski.info/2016/11/01/statistics-and-dynamics)  
  ["Reactive Vs Predictive AI"](http://blog.piekniewski.info/2016/11/03/reactive-vs-predictive-ai)  
  ["Mt. Intelligence"](http://blog.piekniewski.info/2017/11/18/mt-intelligence)  
  ["Learning Physics Is The Way To Go"](http://blog.piekniewski.info/2016/11/30/learning-physics-is-the-way-to-go)  
  ["Predictive Vision In A Nutshell"](http://blog.piekniewski.info/2016/11/04/predictive-vision-in-a-nutshell)  

  ["Common Sense Machine Vision"](https://www.toddhylton.net/2016/10/common-sense-machine-vision.html)

  ["Unsupervised Learning from Continuous Video in a Scalable Predictive Recurrent Network"](https://arxiv.org/abs/1607.06854)  
  ["Fundamental principles of cortical computation: unsupervised learning with prediction, compression and feedback"](https://arxiv.org/abs/1608.06277)  


  *Todd Hylton*

> "The primary problem in computing today is that computers cannot organize themselves: trillions of degrees of freedom doing the same stuff over and over, narrowly focused rudimentary AI capabilities. Our mechanistic approach to the AI problem is ill-suited to complex real-world problems: machines are the sum of their parts and disconnected from the world except through us, the world is not a machine. Thermodynamics drives the evolution of everything. Thermodynamic evolution is the missing, unifying concept in computing systems. Thermodynamic evolution supposes that all organization spontaneously emerges in order to use sources of free energy in the universe and that there is competition for this energy. Thermodynamic evolution is second law of thermodynamics, except that it adds the idea that in order for entropy to increase an organization must emerge that makes it possible to access free energy. The first law of thermodynamics implies that there is competition for energy."

  ["Thermodynamic Computing"](https://youtube.com/watch?v=zFGROl5uPzo)  
  ["On Thermodynamics and the Future of Computing"](https://ieeetv.ieee.org/conference-highlights/on-thermodynamics-and-the-future-of-computing-ieee-rebooting-computing-2017)  
  ["Is the Universe a Product of Thermodynamic Evolution?"](https://youtube.com/watch?v=eB3m4X9xj2A)  
  [Thermodynamic Computing Workshop](https://cccblog.org/2019/02/05/recap-of-the-cccs-thermodynamic-computing-workshop)  

  ["Intelligence is not Artificial"](https://www.toddhylton.net/2016/04/intelligence-is-not-artificial.html)  
  ["Of Men and Machines"](https://www.toddhylton.net/2017/10/of-men-and-machines.html)  


  *Susanne Still*

> "All systems perform computations by means of responding to their environment. In particular, living systems compute, on a variety of length- and time-scales, future expectations based on their prior experience. Most biological computation is fundamentally a nonequilibrium process, because a preponderance of biological machinery in its natural operation is driven far from thermodynamic equilibrium. Physical systems evolve via a sequence of input stimuli that drive the system out of equilibrium and followed by relaxation to a thermal bath."

  ["Optimal Information Processing"](https://youtube.com/watch?v=RrMitURpdq0)  
  ["Optimal Information Processing: Dissipation and Irrelevant Information"](https://youtube.com/watch?v=XCT3ebFqreY)  
  ["Thermodynamic Limits of Information Processing"](https://youtube.com/watch?v=ufywxxu7jww)  

  ["The Thermodynamics of Prediction"](https://arxiv.org/abs/1203.3271)  
  ["An Information-theoretic Approach to Curiosity-driven Reinforcement Learning"](https://researchgate.net/publication/229082289_An_information-theoretic_approach_to_curiosity-driven_reinforcement_learning)  
  ["Information Theoretic Approach to Interactive Learning"](https://arxiv.org/abs/0709.1948)  


  *Karl Friston*

> "The free energy principle seems like an attempt to unify perception, cognition, homeostasis, and action. Free energy is a mathematical concept that represents the failure of some things to match other things they’re supposed to be predicting. The brain tries to minimize its free energy with respect to the world, ie minimize the difference between its models and reality. Sometimes it does that by updating its models of the world. Other times it does that by changing the world to better match its models. Perception and cognition are both attempts to create accurate models that match the world, thus minimizing free energy. Homeostasis and action are both attempts to make reality match mental models. Action tries to get the organism’s external state to match a mental model. Homeostasis tries to get the organism’s internal state to match a mental model. Since even bacteria are doing something homeostasis-like, all life shares the principle of being free energy minimizers. So life isn’t doing four things – perceiving, thinking, acting, and maintaining homeostasis. It’s really just doing one thing – minimizing free energy – in four different ways – with the particular way it implements this in any given situation depending on which free energy minimization opportunities are most convenient."

  ["Free Energy Principle"](https://youtube.com/watch?v=NIu_dJGyIQI)  
  ["Free Energy and Active Inference"](https://youtube.com/watch?v=dLXKFA33SSM)  
  ["Active Inference and Artificial Curiosity"](https://youtube.com/watch?v=Y1egnoCWgUg)  
  ["Active Inference and Artificial Curiosity"](https://youtube.com/watch?v=VHJiTO5ZlYA)  
  ["Uncertainty and Active Inference"](https://slideslive.com/38909755/uncertainty-and-active-inference)  

  [introduction](http://slatestarcodex.com/2018/03/04/god-help-us-lets-try-to-understand-friston-on-free-energy) to free energy minimization  
  [tutorial](https://medium.com/@solopchuk/tutorial-on-active-inference-30edcf50f5dc) on active inference  
  [tutorial](https://medium.com/@solopchuk/free-energy-action-value-and-curiosity-514097bccc02) on free energy and curiosity  
  [implementation](https://kaiu.me/2017/07/11/introducing-the-deep-active-inference-agent)  

  ["The Free-Energy Principle: A Unified Brain Theory?"](https://www.researchgate.net/publication/41001209_Friston_KJ_The_free-energy_principle_a_unified_brain_theory_Nat_Rev_Neurosci_11_127-138)  
  ["Exploration, Novelty, Surprise, and Free Energy Minimization"](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3791848)  
  ["Action and Behavior: a Free-energy Formulation"](https://www.fil.ion.ucl.ac.uk/~karl/Action%20and%20behavior%20A%20free-energy%20formulation.pdf)  
  ["Computational Mechanisms of Curiosity and Goal-directed Exploration"](https://biorxiv.org/content/early/2018/09/07/411272)  
  ["Expanding the Active Inference Landscape: More Intrinsic Motivations in the Perception-Action Loop"](https://arxiv.org/abs/1806.08083)  




### closing words

  Solving many problems in science/engineering might not require computer intelligence described above — if computers will continue to be programmed to solve non-standard problems by humans as it is today. But some very important (and most hyped) problems such as robotics (truly unconstrained self-driving) and language understanding (truly personal assistant) might remain unsolved without such intelligence.
