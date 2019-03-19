  https://habr.com/en/post/436458

---


  The biggest issue with AI is not that it is stupid but lack of definition for intelligence and hence lack of measure for it [[0]](https://arxiv.org/abs/0712.3329) [[1]](https://blog.piekniewski.info/2017/04/13/ai-confuses-intelligent) [[2]](http://blog.piekniewski.info/2016/08/09/intelligence-is-real).

  Turing test is not a good measure because gorilla Koko wouldn't pass but she could solve more problems than lots of disabled human beings [[3]](http://rodneybrooks.com/forai-steps-toward-super-intelligence-ii-beyond-the-turing-test) [[4]](http://rodneybrooks.com/forai-steps-toward-super-intelligence-iv-things-to-work-on-now).

  It is quite possible that people in the future will wonder why so many people back in 2019 thought playing Go and other fixed games in simulated environments after long training had anything to do with intelligence.

  Intelligence is more about adapting/transfering old knowledge to new task (playing Quake Arena quite good without any training after mastering Doom) than it is about compressing experience into heuristics to predict outcome and searching action in given position to maximize predicted outcome value (playing Quake Arena quite good after million games after mastering Doom).

  Human intelligence is about ability to adapt to physical and social world [[4a]](https://amazon.com/Measure-All-Minds-Evaluating-Intelligence/dp/1107153018), and playing Go is a particular adaptation performed by human intelligence, and developing algorithm to learn to play Go is more performant adaptation, and developing mathematical theory to play Go might be even more performant.

  It makes more sense to compare a human and AI not by effectiveness/efficiency of end product of adaptation (in games played between human and agent) but by effectiveness/efficiency of process of adaptation (in games played between human-coded agent and machine-learned agent after limited practice). <cut/>

  Dota 2, StarCraft 2, Civilization 5 and probably even GTA 5 might be solved in not so distant future but ability to play any new game at human level with no prior training would be way more significant.

  The second biggest issue with AI is lack of robustness in a long tail of uncommon cases (and critical ones in medicine, self-driving vehicles, finance) which presently can't be handled with accuracy even close to acceptable [[5]](http://blog.piekniewski.info/2016/11/15/ai-and-the-ludic-fallacy) [[6]](http://blog.piekniewski.info/2017/01/13/outside-the-box) [[7]](http://blog.piekniewski.info/2017/07/27/measuring-performance-ml) [[8]](http://blog.piekniewski.info/2017/03/06/give-me-a-dataset-to-train-and-i-shall-move-the-world) [[9]](https://blog.piekniewski.info/2017/04/03/the-complexity-of-simplicity).

  Complex models exploit any patterns that relate input to output variables but some patterns might not hold for cases poorly covered by training data. >99% of healthcare applications use simple models such as logistic regression with heavily engineered features - for better robustness on outliers [[10a]](https://www.sciencedirect.com/science/article/pii/S0895435618310813) [[10b]](https://www.thelancet.com/action/showPdf?pii=S2589-5370%2819%2930037-9) [[10c]](https://twitter.com/david_sontag/status/1009629318635491330). The clear trend in popular web services is more feature engineering (converting domain knowledge into code to compute statistics, obtaining better performance using more relevant knowledge) - not more sophisticated Deep Learning models (mostly used in production for applications where feature engineering might not lead to better performance such as sensoric applications).

  For some problems such as games in simulated environments like Go one can generate training data using true model such as game rules. Discovering interesting properties of that data isn't intelligent - for most real-world problems discovering true model is the point.

  For an organism, real world is not a fixed game with known environment and rules such as Go or Quake but a game with environment and rules largerly unknown and always changing. It has to adapt to unexpected changes of environment and rules including changes caused by adversaries. It has to be capable of wide autonomy as opposed to merely automation necessary to play some fixed game.

  It might turn out to be impossible to have humanoid robots and self-driving vehicles operating alongside humans without training them to obtain human-level adaptability to real world. It might turn out to be impossible to have personal assistants substituting humans in key aspects of their lives without training them to obtain human-level adaptability to social world [[11]](http://rodneybrooks.com/forai-steps-toward-super-intelligence-ii-beyond-the-turing-test).




### knowledge vs intelligence

  Knowledge is some information, such as observations or experiences, compressed and represented in some computable form, such as text in natural language, mathematical theory in semi-formal language, program in formal language, weights of neural network or synapses of brain.

  Knowledge is about tool (theory, algorithm, physical process) to solve problem. Intelligence is about applying (transfering) and creating (learning) knowledge. There is knowledge how to solve problem (algorithm for computer, instructions for human), and then there is process of applying knowledge (executing program by computer, interpreting and executing instructions by human), and then there is process of creating knowledge (inductive inference/learning from observations and experiments, deductive reasoning from inferred theories and learned models).

  Alpha(Go)Zero is way closer to knowledge how to solve particular class of problems than to an intelligent agent capable of applying and creating knowledge. It is a search algorithm like IBM Deep Blue with heuristics being not hardcoded but being tuned during game sessions. It can't apply learned knowledge to other problems - even playing on smaller Go board. It can't create abstract knowledge useful to humans - even simple insight on Go tactics.

  TD-Gammon from 1992 is considered by many as the biggest breakthrough in AI [[12]](https://youtube.com/watch?v=9EN_HoEk3KY&t=29m44s). Note that TD-Gammon didn't use Q-learning - it used TD(λ) with online on-policy updates. TD-Gammon's author used its variation to learn IBM Watson's wagering strategy [[13]](https://youtube.com/watch?v=7rIf2Njye5k).

  Alpha(Go)Zero is also roughly a variation of TD(λ) [[14]](http://techtalks.tv/talks/on-td-learning-and-links-with-deeprl-in-atari-and-alphago/63031). TD-Gammon used neural network trained by Temporal Difference learning with target values calculated using tree search with depth not more than three and using results of full game playouts as estimates of leaf values. Alpha(Go)Zero used deep neural network trained by Temporal Difference learning with target values calculated using Monte-Carlo Tree Search with much bigger depth and using estimates of leaf values and tree policy calculated by network without full game playouts.

  Qualitative differences between Backgammon and Go as problems and between TD-Gammon and Alpha(Go)Zero as solutions (scale of neural network and number of played games being major differences) are not nearly as big as qualitative differences between perfect information games such as Go and imperfect information games such as Poker (Alpha(Go)Zero and Libratus can't be used correspondingly for Poker and Go).

  IBM Watson, the most advanced question answering system by far in 2011, is not an intelligent agent. It is knowledge represented as 100Ks lines of hand-crafted logic for searching / manipulating sequences of words and generating hypotheses / supporting arguments plus few hundred parameters tuned with linear regression for weighing in different pieces of knowledge for each supported type of question / answer [[15]](https://youtube.com/watch?v=DywO4zksfXw) [[16]](https://youtu.be/1n-cwezu8j4?t=53m4s). It's not that much different conceptually from database engines which use statistics of data and hardcoded threshold values to construct a plan for executing given query via selecting and pipelining a subset of implemented algorithms for manipulating data.

  Google Duplex must be a heavily engineered solution for very narrow domains and tasks which involves huge amount of human labor to write rules and to label data. Google was reported to employ 100 of PhD linguists working just on rules and data for question answering in Google Search and Assistant [[17]](https://wired.com/2016/11/googles-search-engine-can-now-answer-questions-human-help).

  IBM Debater is a heavily engineered solution for finding and summarizing texts relevant to given topic (without state-of-the-art results on academic benchmarks for summarization) [[18]](https://scholar.google.com/citations?user=KjvrNGMAAAAJ&sortby=pubdate) [[19]](https://youtube.com/watch?v=TYJ3iwW59_w) [[20]](https://youtube.com/watch?v=8Xd4-cV9d74). It can't answer opponent's arbitrary questions about its own arguments because it neither has or learns any model of domain it argues about.




### what is intelligence

  Biologists define intelligence as ability to find non-standard solutions for non-standard problems and distinguish it from reflexes and instincts defined as standard solutions for standard problems. Playing Go can't be considered a non-standard problem for AlphaGo after playing millions of games. Detecting new malware can be considered a non-standard problem with no human-level solution so far.

  Necessity to adapt/survive provides optimization objectives for organisms to guide self-organization and learning/evolution. Some organisms can set up high-level objectives for themselves after being evolved/trained to satisfy low-level objectives.

  Most AI researchers focus on top-down approach to intelligence, i.e. defining objective for high-level problem (such as maximizing expected probability of win by Alpha(Go)Zero) and expecting agent to learn good solutions for low-level subproblems. This approach works for relatively simple problems like games in simulated environments but requires enormous amount of training episodes (several orders of magnitude more than amount which can be experienced by agent in real world) and leads to solutions incapable of generalization (Alpha(Go)Zero trained on 19x19 board which can't be adapted to 9x9 board without retraining). Hardest high-level problems which can be solved by humans are open-ended - humans don't search in fixed space of possible solutions unlike AlphaGo. Being informed and guided by observations and experiments in real world, humans come up with good subproblems, e.g. special and general relativity.

  A few AI researchers [section "possible directions"] focus on bottom-up approach, i.e. starting with low-level objectives (such as maximizing predictability of environment or of effect on environment), then adding higher-level objectives for intrinsic motivation (such as maximizing learning progress or available future options), and only then adding high-level objectives for problems of interest (such as maximizing game score). This approach is expected to lead to more generalizable and robust solutions for high-level problems because learning with low-level objectives leads to agent also learning self-directing and self-correcting behavior helpful in non-standard or dangerous situations with zero information about them effectively provided by high-level objective. It is quite possible that some set of universal low-level objectives might be derived from a few equations governing flow of energy and information, so that optimization with those objectives might lead to intelligence of computers in a analogous way to how evolution of the Universe governed by laws of physics leads to intelligence of organisms.

  While solving high-level problems in simulated environments such as Go had successes, solving low-level problems such as vision and robotics are yet to have such successes. Humans can't learn to play Go without first learning to discern board and move stones. Computer can solve some high-level problems without ability to solve low-level ones when high-level problems are abstracted away from low-level subproblems by humans. It is low-level problems which are more computationally complex for both humans and computers although not necessarily more complex as mathematical or engineering problems. It is low-level problems which are prerequisites for commonsense reasoning, i.e. estimating plausibility of given hypothesis from given observations and previously acquired knowledge, which is necessary for machine to adapt to arbitrary given environment and to solve arbitrary high-level problem in that environment [[21]](https://en.wikipedia.org/wiki/Moravec%27s_paradox) [[22]](https://youtu.be/DIE0jG1Jk8k?t=59m23s).
 
  The biggest obstacle to applications in real-world environments as opposed to simulated environments seems to be underconstrained objectives for optimization. Any sufficiently complex model trained with insufficiently constrained objective will exploit any pattern found in training data that relates input to target variables but spurious correlations won't generalize to testing data [[23]](http://blog.piekniewski.info/2017/03/06/give-me-a-dataset-to-train-and-i-shall-move-the-world) [[24]](https://en.wikipedia.org/wiki/Goodhart%27s_law). Even billion examples don't constrain optimization sufficiently and don't lead to major performance gains in image recognition [[25]](https://arxiv.org/abs/1707.02968) [[26]](https://youtube.com/watch?v=9497PCSNss4). Agent finds surprising ways to exploit simulated environment to maximize objective not constrained enough to prevent exploits [[27]](https://youtube.com/watch?v=-p7VhdTXA0k).

  Two possible ways to constrain optimization sufficiently to avoid non-generalizable and non-robust solutions are more informative data for training (using physics of real world or dynamics of social world as sources of signal as opposed to simulated environments not nearly as complex and not representative of corner cases in real/social world) and more complex objectives for optimization (predicting not only statistics of interest such as future cumulative rewards conditionally on agent's next actions but also predicting dynamics, i.e. some arbitrary future properties of environment conditionally on some arbitrary hypothetical future events including agent's next actions - progress in learning to make such predictions might be the strongest form of intrinsic motivation for agent).

  There is enormous gap between complexity of simulated environments available for present computers and complexity of real-world environments available for present robots so that agent trained in simulated environment can't be transfered to robot in real-world environment with acceptable performance and robustness [[28]](https://youtube.com/watch?v=7A_QPGcjrh0). Boston Dynamics team never used machine learning to control their robots - they use real-time solvers of differential equations to calculate dynamics and optimal control for models of robots and environments which are not learned from data but specified manually [[29]](https://youtube.com/watch?v=aFuA50H9uek) [[30]](https://quora.com/What-kind-of-learning-algorithms-are-used-on-Boston-Dynamics-robots/answer/Eric-Jang). MIT researchers didn't use machine learning to control their robot in DARPA Robotics Challenge 2015, and their robot was the only robot which didn't fall or need physical assistance from humans [[31]](https://youtube.com/watch?v=2GW7ozcUCFE).

  Long tails might be not learnable statistically and require reasoning at test time. It might be impossible to encode all patterns into parameters of statistical model. A phenomena might not admit separating data by any number of decision hyper-planes. Not only statistics but dynamics of phenomena might have to be calculated by model. Model might have to be programmed or/and trained to simulate dynamics of phenomena.

  It's quite possible that the only way to train/evolve intelligent agent for hard problems in real world (such as robotics) and in social world (such as natural language understanding) might turn out to be:  
  (1) to train/evolve agent in environment which provides as much constraints for optimization as real and social world (i.e. agent has to be a robot operating in real world alongside humans);  
  (2) to train/evolve agent on problems which provide as much constraints for optimization as the hardest problems solved by organisms in real world (i.e. robot has to learn to survive without any assistance from humans) and solved by humans in social world (i.e. agent has to learn to reach goals in real world using conversations with humans as its only tool).




### progress

  Arguably during Deep Learning renaissance period there hasn't been progress in real-world problems such as robotics, language understanding, healthcare nearly as significant as in fixed games running in simulated environments.

  Opinions on progress of AI research from some of the most realistic researchers:  

  *Michael I. Jordan*  [[32]](https://medium.com/@mijordan3/artificial-intelligence-the-revolution-hasnt-happened-yet-5e1d5812e1e7) [[33]](https://youtu.be/4inIBmY8dQI?t=13m9s) [[34]](https://youtu.be/Tkl6ERLWAbA?t=3m17s)  
  *Rodney Brooks*  [[35]](https://rodneybrooks.com/forai-steps-toward-super-intelligence-i-how-we-got-here) [[36]](http://rodneybrooks.com/the-seven-deadly-sins-of-predicting-the-future-of-ai)  
  *Philip Piekniewski*  [[37]](http://blog.piekniewski.info/2016/08/02/80p_there) [[38]](https://blog.piekniewski.info/2018/06/20/rebooting-ai-postulates)  
  *Francois Chollet*  [[39]](https://blog.keras.io/the-limitations-of-deep-learning.html) [[40]](https://medium.com/@francois.chollet/the-impossibility-of-intelligence-explosion-5be4a9eda6ec)  
  *John Langford*  [[41]](http://hunch.net/?p=9828091) [[42]](http://hunch.net/?p=9604328)  
  *Alex Irpan*  [[43]](https://www.alexirpan.com/2018/02/14/rl-hard.html)  


  Deep Learning methods are very non-robust in image understanding tasks [[44]](https://thegradient.pub/the-limitations-of-visual-deep-learning-and-how-we-might-fix-them) [[45]](https://youtube.com/watch?v=9497PCSNss4) [[46]](https://blog.piekniewski.info/2016/08/12/how-close-are-we-to-vision) [[47]](https://blog.piekniewski.info/2016/08/18/adversarial-red-flag) [[48]](https://blog.piekniewski.info/2016/12/29/can-a-deep-net-see-a-cat) [[49]](https://petewarden.com/2018/07/06/what-image-classifiers-can-do-about-unknown-objects).  
  Deep Learning methods haven't come even close to replacing radiologists [[50]](https://twitter.com/lpachter/status/999772075622453249) [[51]](https://twitter.com/zacharylipton/status/999395902996516865) [[52]](https://medium.com/@jrzech/what-are-radiological-deep-learning-models-actually-learning-f97a546c5b98).  
  Deep Learning methods are very non-robust in text understanding tasks [[53]](https://thegradient.pub/frontiers-of-generalization-in-natural-language-processing) [[54]](https://motherboard.vice.com/en_us/article/j5npeg/why-is-google-translate-spitting-out-sinister-religious-prophecies).  
  Deep Learning methods can't solve questions from school science tests significantly better than text search based methods [[55]](http://data.allenai.org/arc/challenge-train) [[56]](https://arxiv.org/abs/1803.05457).  
  Deep Learning methods can't pass first levels of the hardest Atari game - it requires abstracting and reasoning from agent [[57]](https://youtube.com/watch?v=_zbg9rs5QZY) [[58]](https://medium.com/@awjuliani/on-solving-montezumas-revenge-2146d83f0bc3).  

  ["Measuring the Tendency of CNNs to Learn Surface Statistical Regularities"](https://arxiv.org/abs/1711.11561)  
  ["Approximating CNNs with Bag-of-local-Features Models Works Surprisingly Well on ImageNet"](https://openreview.net/forum?id=SkfMWhAqYQ)  
  ["Do ImageNet Classifiers Generalize to ImageNet?"](http://people.csail.mit.edu/ludwigs/papers/imagenet.pdf)  
  ["Confounding Variables Can Degrade Generalization Performance of Radiological Deep Learning Models"](https://arxiv.org/abs/1807.00431)  
  ["One Pixel Attack for Fooling Deep Neural Networks"](https://arxiv.org/abs/1710.08864)  
  ["A Rotation and a Translation Suffice: Fooling CNNs with Simple Transformations"](https://arxiv.org/abs/1712.02779)  
  ["Semantic Adversarial Examples"](https://arxiv.org/abs/1804.00499)  
  ["Why Do Deep Convolutional Networks Generalize so Poorly to Small Image Transformations?"](https://arxiv.org/abs/1805.12177)  
  ["The Elephant in the Room"](https://arxiv.org/abs/1808.03305)  
  ["Strike (with) a Pose: Neural Networks Are Easily Fooled by Strange Poses of Familiar Objects"](https://arxiv.org/abs/1811.11553)  
  ["Semantically Equivalent Adversarial Rules for Debugging NLP models"](https://acl2018.org/paper/1406)  
  ["On GANs and GMMs"](https://arxiv.org/abs/1805.12462)  
  ["Do Deep Generative Models Know What They Don't Know?"](https://arxiv.org/abs/1810.09136)  
  ["Are Generative Deep Models for Novelty Detection Truly Better?"](https://arxiv.org/abs/1807.05027)  
  ["Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations"](https://arxiv.org/abs/1811.12359)  
  ["Simple Random Search Provides a Competitive Approach to Reinforcement Learning"](https://arxiv.org/abs/1803.07055)  




### possible directions

  *Juergen Schmidhuber*

> "Data becomes temporarily interesting by itself to some self-improving, but computationally limited, subjective observer once he learns to predict or compress the data in a better way, thus making it subjectively simpler and more beautiful. Curiosity is the desire to create or discover more non-random, non-arbitrary, regular data that is novel and surprising not in the traditional sense of Boltzmann and Shannon but in the sense that it allows for compression progress because its regularity was not yet known. This drive maximizes interestingness, the first derivative of subjective beauty or compressibility, that is, the steepness of the learning curve. It motivates exploring infants, pure mathematicians, composers, artists, dancers, comedians, yourself, and artificial systems."

  ["The Simple Algorithmic Principle behind Creativity, Art, Science, Music, Humor"](https://youtube.com/watch?v=h7F5sCLIbKQ)  
  ["Formal Theory of Fun and Creativity"](http://videolectures.net/ecmlpkdd2010_schmidhuber_ftf)  

  ["Driven by Compression Progress: A Simple Principle Explains Essential Aspects of Subjective Beauty, Novelty, Surprise, Interestingness, Attention, Curiosity, Creativity, Art, Science, Music, Jokes"](http://arxiv.org/abs/0812.4360)  
  ["Formal Theory of Creativity, Fun, and Intrinsic Motivation"](http://people.idsia.ch/~juergen/ieeecreative.pdf)  


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

  ["Unsupervised Learning from Continuous Video in a Scalable Predictive Recurrent Network"](https://arxiv.org/abs/1607.06854)


  *Todd Hylton*

> "The primary problem in computing today is that computers cannot organize themselves: trillions of degrees of freedom doing the same stuff over and over, narrowly focused rudimentary AI capabilities. Our mechanistic approach to the AI problem is ill-suited to complex real-world problems: machines are the sum of their parts and disconnected from the world except through us, the world is not a machine. Thermodynamics drives the evolution of everything. Thermodynamic evolution is the missing, unifying concept in computing systems. Thermodynamic evolution supposes that all organization spontaneously emerges in order to use sources of free energy in the universe and that there is competition for this energy. Thermodynamic evolution is second law of thermodynamics, except that it adds the idea that in order for entropy to increase an organization must emerge that makes it possible to access free energy. The first law of thermodynamics implies that there is competition for energy."

  ["On Thermodynamics and the Future of Computing"](https://ieeetv.ieee.org/conference-highlights/on-thermodynamics-and-the-future-of-computing-ieee-rebooting-computing-2017)  
  ["Is the Universe a Product of Thermodynamic Evolution?"](https://youtube.com/watch?v=eB3m4X9xj2A)  
  ["Thermodynamic Computing Workshop"](https://cccblog.org/2019/02/05/recap-of-the-cccs-thermodynamic-computing-workshop)  

  ["Fundamental principles of cortical computation: unsupervised learning with prediction, compression and feedback"](https://arxiv.org/abs/1608.06277)


  *Karl Friston*

> "The free energy principle seems like an attempt to unify perception, cognition, homeostasis, and action. Free energy is a mathematical concept that represents the failure of some things to match other things they’re supposed to be predicting. The brain tries to minimize its free energy with respect to the world, ie minimize the difference between its models and reality. Sometimes it does that by updating its models of the world. Other times it does that by changing the world to better match its models. Perception and cognition are both attempts to create accurate models that match the world, thus minimizing free energy. Homeostasis and action are both attempts to make reality match mental models. Action tries to get the organism’s external state to match a mental model. Homeostasis tries to get the organism’s internal state to match a mental model. Since even bacteria are doing something homeostasis-like, all life shares the principle of being free energy minimizers. So life isn’t doing four things – perceiving, thinking, acting, and maintaining homeostasis. It’s really just doing one thing – minimizing free energy – in four different ways – with the particular way it implements this in any given situation depending on which free energy minimization opportunities are most convenient."

  ["Free Energy Principle"](https://youtube.com/watch?v=NIu_dJGyIQI)  
  ["Free Energy and Active Inference"](https://youtube.com/watch?v=dLXKFA33SSM)  
  ["Active Inference and Artificial Curiosity"](https://youtube.com/watch?v=Y1egnoCWgUg)  
  ["Active Inference and Artificial Curiosity"](https://youtube.com/watch?v=VHJiTO5ZlYA)  
  ["Uncertainty and Active Inference"](https://slideslive.com/38909755/uncertainty-and-active-inference)  

  [introduction](http://slatestarcodex.com/2018/03/04/god-help-us-lets-try-to-understand-friston-on-free-energy) to free energy minimization  
  [tutorial](https://medium.com/@solopchuk/tutorial-on-active-inference-30edcf50f5dc) on active inference  

  ["The Free-Energy Principle: A Unified Brain Theory?"](https://www.researchgate.net/publication/41001209_Friston_KJ_The_free-energy_principle_a_unified_brain_theory_Nat_Rev_Neurosci_11_127-138)  
  ["Action and Behavior: a Free-energy Formulation"](https://www.fil.ion.ucl.ac.uk/~karl/Action%20and%20behavior%20A%20free-energy%20formulation.pdf)  
  ["Computational Mechanisms of Curiosity and Goal-directed Exploration"](https://biorxiv.org/content/early/2018/09/07/411272)  
  ["Expanding the Active Inference Landscape: More Intrinsic Motivations in the Perception-Action Loop"](https://arxiv.org/abs/1806.08083)  


  *Alex Wissner-Gross*

> "Intelligent system needs to optimize future causal entropy, or to put it in plain language, maximize the available future choices. Which in turn means minimizing all the unpleasant situations with very few choices. This makes sense from evolutionary point of view as it is consistent with the ability to survive, it is consistent with what we see among humans (collecting wealth and hedging on multiple outcomes of unpredictable things) and generates reasonable behavior in several simple game situations."

  ["An Equation for Intelligence"](https://youtube.com/watch?v=PL0Xq0FFQZ4)  
  ["The Physics of Artificial General Intelligence"](https://youtube.com/watch?v=dIUq3LYLEQo)  

  ["Causal Entropic Forces"](http://math.mit.edu/~freer/papers/PhysRevLett_110-168702.pdf)


  *Susanne Still*

> "All systems perform computations by means of responding to their environment. In particular, living systems compute, on a variety of length- and time-scales, future expectations based on their prior experience. Most biological computation is fundamentally a nonequilibrium process, because a preponderance of biological machinery in its natural operation is driven far from thermodynamic equilibrium. Physical systems evolve via a sequence of input stimuli that drive the system out of equilibrium and followed by relaxation to a thermal bath."

  ["Optimal Information Processing"](https://youtube.com/watch?v=RrMitURpdq0)  
  ["Optimal Information Processing: Dissipation and Irrelevant Information"](https://youtube.com/watch?v=XCT3ebFqreY)  
  ["Thermodynamic Limits of Information Processing"](https://youtube.com/watch?v=ufywxxu7jww)  

  ["The Thermodynamics of Prediction"](https://arxiv.org/abs/1203.3271)




### closing words

  Solving many problems in science and engineering may not need computer intelligence defined above - if computers will be programmed to solve non-standard problems by humans as it is done today. But some very important (and most hyped) problems such as robotics (truly unconstrained self-driving) and natural language understanding (truly personal assistant) might not admit sufficient progress without such intelligence.
