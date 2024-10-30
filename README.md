# Research Paper Distillation By Zhenghao Chi

> To see paper recommendation level and tags, visit the doc on Lark [here](https://ndro4zkb6p.feishu.cn/docx/FaBXdERDvoleXDxSiMXcMBegnvd?from=from_copylink).

# Robotics

### HumanPlus: Humanoid Shadowing and Imitation from Humans

> Fu, Z., Zhao, Q., Wu, Q., Wetzstein, G., & Finn, C. (2024). HumanPlus: Humanoid Shadowing and Imitation from Humans. _arXiv preprint arXiv:2406.10454_.

- General Idea:

  1. **X-embodiment Shadow policy**: RGB third-vision human bahaviour -> humanoid bahaviour
  2. Data collection for behaviour cloning: Use shadow policy + human demonstration, humanoid get **egocentric binocular RGB camera** data of conducting a task.
- Some technical points

  1. During shadow policy training, the DoF of robot and human are diff. Need to map between paramatrization of these two. Then they can train policy using RL.
  2. RL policy inputs humanoid proprioception and a humanoid target pose; outputs target joint position&velocity. Then they have a PD controller to get torque.
- Limitations

  1. Fixed mapping between different DoF parametrization
  2. Error can be cascaded -> error of pose estimation & RL policy & testing error
  3. No long-horizon task because no large size of human demonstration
- Utilities

  1. Human pose paramatrization model: [SMPL-X](https://smpl-x.is.tue.mpg.de)
  2. Human body pose sequence dataset: [AMASS](https://amass.is.tue.mpg.de)
  3. State-of-art body & hand pose estimation model: need survey

### GNM: A General Navigation Model to Drive Any Robot

> Shah, D., Sridhar, A., Bhorkar, A., Hirose, N., & Levine, S. (2023, May). Gnm: A general navigation model to drive any robot. In _2023 IEEE International Conference on Robotics and Automation (ICRA)_ (pp. 7226-7233). IEEE.

- General Idea:

  1. Image-goal navigation (image observation of the goal position) to allow general formulation of the task
  2. Hand-crafted abstract action representation as policy output (execution depends on controller)
  3. Learnt embodiment context vector using consecutive past observations from robot's viewpoint and condition the policy on it.
- Limitation:

  1. Hand-craft action representation
  2. The embodiment vector mostly contains info of velocity,size, etc. Not 'what it can do' or affordance. So no 'brand-new' navigation strategy emerged tailored to the capabilities of new embodiments.

### Neural MP: A Generalist Neural Motion Planner

> Dalal, M., Yang, J., Mendonca, R., Khaky, Y., Salakhutdinov, R., & Pathak, D. (2024). Neural MP: A Generalist Neural Motion Planner. _arXiv preprint arXiv:2409.05864_.

- General Idea

  1. Diverse scene generation in simulation and use 3D object datasets
  2. Distill motion planning via visual imiation learning
  3. Test time optimization to enhance safety
  4. Just marginal improvement of perf and largely the same as [MπNets](https://proceedings.mlr.press/v205/fishman23a.html).
- Utilities

  1. Large 3D obj dataset [Objaverse](https://objaverse.allenai.org)

### MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations

> Mandlekar, A., Nasiriany, S., Wen, B., Akinola, I., Narang, Y., Fan, L., ... & Fox, D. (2023). Mimicgen: A data generation system for scalable robot learning using human demonstrations. _arXiv preprint arXiv:2310.17596_.

- General Idea

  1. Data augmentation paper: generate more data from a few human demonstrations
  2. Simple learning algorithm and hardware: naive behavior cloning + 1 * V100
  3. Tasks must be divisible into **object-centric subtasks** (see appedix K)
  4. Linear interpolation between each subtask
  5. Use $T^A_B$ frame transformation to generate new data with **fixed gripper-object relative pose**
- Limitation

  1. Object-centric subtask assumption (and exactly one obj per subtask)
  2. Naive data filtering (success or not)
  3. No guarantee on collision-free motion (especially during interpolation)
  4. Objects should be rigid-body and new data only has similar objects as source data
  5. Objects and robot bases can't be dynamic
  6. Can't support multi-arm tasks
- Utility

  1. Some related literatures in appendix: Full related work section & motivation for mimicgen over alternative methods section

### Contrastive Representation Learning

> Weng, Lilian. (May 2021). Contrastive representation learning. Lil’Log. [https://lilianweng.github.io/posts/2021-05-31-contrastive/](https://lilianweng.github.io/posts/2021-05-31-contrastive/).

- Key points

  1. Hard negative samples are needed to improve the model
  2. What does it mean by 'hard': different label but close in embedding space (thus need to 'drag apart')
  3. Large batch size is typically needed, so the data is diverse
- Utility

  1. [Blog post on explaining BYOL and intuition of why contrastive learning](https://imbue.com/research/2020-08-24-understanding-self-supervised-contrastive-learning/): Batch normalization implicitly introduce contrastive learning in BYOL
  2. Visual data augmentation methods can be checked
  3. Works on loss function design

### Robot Learning on the Job: Human-in-the-Loop Autonomy and Learning During Deployment

> Liu, H., Nasiriany, S., Zhang, L., Bao, Z., & Zhu, Y. (2022). Robot learning on the job: Human-in-the-loop autonomy and learning during deployment. _arXiv preprint arXiv:2211.08416_.

- General Idea

  1. Enable human intervention during policy deployment to continuously improve the policy
  2. Weight behavior cloning by data quality approximated by two criteria: First, human interventions are high quality and should be learnt immedietely; Second, trajectory right before intervention is of bad quality
  3. Label data using classes: demo(initial human demo data), intervention, prev-intervention, robot
  4. Memory management strategy: LFI - first reject samples from trajectories with the least interventions
- Limitation

  1. Requires the human to constantly monitor the robot. Should incorporate automated runtime monitoring and error detection strategies.
  2. Experimental tasks only contain insertion. This is ideal for human intervention, but class labels may have different relative importance in other types of tasks
- Interesting Points

  1. LFI memory management strategy outperforms the 'keep all' strategy. This is totally a **dataset distillation** method. Also, high quality data may be under-learnt when data size explodes because batch size is limited

### RoboCook: Long-Horizon Elasto-Plastic Object Manipulation with Diverse Tools

> Shi, H., Xu, H., Clarke, S., Li, Y., & Wu, J. (2023). Robocook: Long-horizon elasto-plastic object manipulation with diverse tools. _arXiv preprint arXiv:2306.14447_.

- General Idea

  1. Perception: Model surface shape change of elasto-plastic stuff with PointCloud and Graph
  2. Action parametrization: Parametrize tool usage action to reduce DoF
  3. Model interaction with elasto-plastic stuff: Goal is to predict how the shape of elasto-plastic stuff will change give an action with a tool. Data collected by robot randomly interact with elasto-plastic stuff and minimize distant between predicted graph and real graph. The result is GNN.
  4. Self-supervised optimal action policy: Given a,b,c, train a policy to get an action to transfer elasto-plastic stuff between states. Data collected in c is reusable by giving action label and states before&after this action. The policy learns by predicting the action label
  5. Closed-Loop Control: Given a,b,c,d, choosing tool that can achieve state closest to the goal state -> Do the action -> Get next state -> next loop
- Limitation

  1. So many learning techniques to achieve Best System Paper Award, CoRL 2023
- Interesting Point

  1. Describe interaction with elasto-plastic stuff with GNN + Point Cloud
  2. **Self-supervised policy design enables cheap and fast real-world data collection pipeline**
  3. Inductive bias of GNN. Train on predicting 2 steps, inference on predicting 15 steps
  4. Achieved **ultra fast planning speed** compared with previous work
- Utility

  1. Lots of GNN+PointCloud techniques
  2. Writing style: Divide sections by problems to be solved and start each section from a problem

### The Plenoptic Function and the Elements of Early Vision

> Adelson, E. H., & Bergen, J. R. (1991). _The plenoptic function and the elements of early vision_ (Vol. 2). Cambridge, MA, USA: Vision and Modeling Group, Media Laboratory, Massachusetts Institute of Technology.

- General Idea
  1. The whole scene can be represented by a plenoptic function whose parameters are properties that can be directly measured by retina ( $x,y,t,\lambda,V_x,V_y,V_z$ in the paper - $x,y$ are point on a plane in front of the eye position, $t$ is time, $\lambda$ is wavelength or color, $V_x,V_y,V_z$ are position of the eye)
  2. Creatures **sample** viewpoint in some ways (i.e: moving head) to get an instance of input of the plenoptic function
  3. A sample (viewpoint) can be viewed as chunked Taylor expansion - direct measurement of the local properties of a plenoptic function

### CyberDemo: Augmenting Simulated Human Demonstration for Real-World Dexterous Manipulation

> Wang, J., Qin, Y., Kuang, K., Korkmaz, Y., Gurumoorthy, A., Su, H., & Wang, X. (2024). CyberDemo: Augmenting Simulated Human Demonstration for Real-World Dexterous Manipulation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ (pp. 17952-17963).

- Main Result
  1. Human demonstration in simulation + augmentation in simulation + fine-tune with real-world demonstration can work.

### AnySkin: Plug-and-play Skin Sensing for Robotic Touch

> Bhirangi, R., Pattabiraman, V., Erciyes, E., Cao, Y., Hellebrekers, T., & Pinto, L. (2024). AnySkin: Plug-and-play Skin Sensing for Robotic Touch. _arXiv preprint arXiv:2409.08276_.

- Main Result

  1. Improve consistency between touch sensor instances
  2. Tactile-only and LSTM-based slip detection policy achieves 92% accuracy on untouched objects
  3. [Open-sourced hardware](https://any-skin.github.io)
- Potential Future Work

  1. More experiment on how much instance-consistency is needed for tactile sensor
  2. The slippery detection policy is trained on human labelled slipped-or-not data. May check more convenient data collection methods and how to integrate them into normal data collection loops (i.e supervised by visual input and improve itself on-the-fly).
  3. Tactile algorithm and representation that facilitates truly hard tasks like cloth folding

### A Tutorial on Energy-Based Learning

> LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M., & Huang, F. (2006). A tutorial on energy-based learning. _Predicting structured data_, _1_(0).

- Main Concepts
  1. The energy-based model learns an energy function $E_\theta(X,Y)$ so that at inference time, given the input $X$ , $argmin_{y\in \mathbb{Y}} E_\theta(X,y)$ is the output. So optimization is needed for inference.
  2. For training, the goal is 'pull up' $E_\theta(X,Y)$ of $y_{wrong}$ and 'pull down' $E_\theta(X,Y)$ of $y_{right}$ . There can be multiple $y_{right}$ .
  3. An idea of loss function is to punish those offending outputs. Offending outputs are $y$ 's that have very low energy so that can be easily detected as good ones. $Loss = E_\theta(X^i,Y^i)+\frac{1}{\beta}log(\int_{y\in\mathbb{Y}}exp\left[-\beta E_\theta(X^i,y)\right])$ is acceptable because $\frac{\partial Loss}{\partial \theta} = \frac{\partial E_\theta(X^i,Y^i)}{\partial \theta} - \int_{y\in\mathbb{Y}} \frac{\partial E_\theta(X^i,y)}{\partial \theta}*P_{\theta}(y|X^i)$ where $P_\theta(y|X^i) = \frac{e^{-\beta E_\theta(X^i,y)}}{\int_{y\in\mathbb{Y}}e^{-\beta E_\theta(X^i,y)}}$ .  So the implication of this loss is to pull down the right answer and meanwhile pull up wrong answers proportion to their 'offending level'.
  4. $\int_{y\in\mathbb{Y}}e^{-\beta E_\theta(X^i,y)}$ is intractable. This causes instability of training when estimated with negative sampling (i.e: the implicit BC paper). **But the idea of this contrastive loss is inspiring**!!! This is solved by only estimating gradient of energy function $w.r.t $ output variable $\frac{\partial E(X^i,y)}{\partial y} = F_\theta(y|X^i)$ , which is the idea of diffusion models.

### OpenVLA: An Open-Source Vision-Language-Action Model

> Kim, M. J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., ... & Finn, C. (2024). OpenVLA: An Open-Source Vision-Language-Action Model. _arXiv preprint arXiv:2406.09246_.

- Main Idea

  1. Fine-tune a visual-language model with visual encoder + projector + LLM backbone with only one image input using the Open-X dataset.
- More questions

  1. SigLIP-DinoV2 visual encoder 'gives spatial understanding'. Check this paper: "Prismatic vlms: Investigating the design space of visually-conditioned language models".
  2. Fine-tuning visual encoder together, rather than freezing it is essential for the perf. The question is what is the difference between the initial visual representation and the fine-tuned representation. Check out.

### HOVER: Versatile Neural Whole-Body Controller for Humanoid Robots

- Main Idea

  1. Unify the command space of humanoid robots with various downstream tasks
- Methods

  1. Train goal-oriented behavior tracking oracle policy and distill. Add mask if the downstream policy is only considered on a subspace of the unified command space.
- My thoughts

  1. Experiments show it can outperform specified command space or method.
  2. **The interesting question is how information from one subspace can help the representation of action in another subspace and thus form a more coordinated whole body control. Can we do self-supervised motion prediction and generation in the unified command space**??
  3. It is still embodiment-specific. The key is it unifies command space of a single embodiment configuration regardless of the downstream tasks.
  4. Many works intentionally avoid high-dim command space by introducing 'end-effector tracking'. Also, high-dim command space is hard for imitation learning. **How to incorporate the unified command space with imitation learning pipeline**?

### Steering Your Generalists: Improving Robotic Foundation Models via Value Guidance

- Main Idea

  1. Test-time search for robots: Re-rank different actions proposed by a pre-trained policy through offline RL
  2. Pre-trained policy: robot doing task following language command $\pi(a|s_t,l)$ . Dataset: $\{\tau_i,l_i\}_{i=1}^N$ where the former is robot action trajectory and the latter is the language command.
  3. Re-rank: sample $K$ actions and $a_t \sim \text{Softmax} \left( \frac{Q_{\theta}(s_t, a_1)}{\beta}, \dots, \frac{Q_{\theta}(s_t, a_K)}{\beta} \right)$ . When $\beta \rightarrow 0$ , the algo is more and more greedier.
- Main results

  1. Very big experiment improvement on some tasks!!
  2. $K=100$ needs inference time 0.15 second. “Compute-optimal” balance between using the policy and querying the value function is needed. Intuition is the value function query can be done in a more high-level way. **The key question is "on what level should we search" for robots**?
  3. "Scaling up value function architectures and using more diverse data is a promising direction for future work".
- Utility

  1. Reward crafting method from arbitrary dataset: "Pre-training for robots: Offline rl enables learning new tasks from a handful of trials".
  2. New value funciton estimation algo: Cal-QL
- TODO

  1. **Do experiment on Cal-QL and re-implement this paper.**

# Fundamental Research of AI

### Were RNNs All We Needed?

> Feng, L., Tung, F., Ahmed, M. O., Bengio, Y., & Hajimirsadegh, H. (2024). Were RNNs All We Needed?. _arXiv preprint arXiv:2410.01201_.

- Main hypothesis

  - The success of Mamba and S4 is because that the proposed hidden state $\hat{h_t}$ does not depend on the past state. That is, the key bone of those RNN models is: $z_t=Z(x_t)$ , $h_{t+1}=(1-z_t)*h_t+z_t*h(x_{t})=(1-z_t)*h_t+z_t*\hat{h_t}$ . Can we achieve similar efficiency and performance using this minimal key bone?
- Main result

  - Very weak experiment because the tasks are extremely easy and tailored to the author's model.
  - Reasonable hypothesis. And the model may be useful in specific tasks that are OK with the model's downsides.
  - Keybone needs more layers because although the first layer capture little global info, upper layers can do that. (trade-off)

### Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning

> Nakamoto, M., Zhai, S., Singh, A., Sobol Mark, M., Ma, Y., Finn, C., ... & Levine, S. (2024). Cal-ql: Calibrated offline rl pre-training for efficient online fine-tuning. _Advances in Neural Information Processing Systems_, _36_.

- Main Points
  1. Scenario is policy initialization using offline RL trained on large dataset and fine-tune on downstream tasks
  2. Previous CQL has an 'unlearning' phenomenon. Since CQL tends to learn a Q-function $\hat{Q}$ much smaller than true value, at the beginning of the fini-tuning, any exploration can get a better return than $\hat{Q}$ of even the learned best action. Thus, the updated $\hat{Q}$ will have highest value at exploration action and thus 'unlearn' the learned policy.
  3. The solution is given by calibration. Intuition is to keep $\hat{Q}$ on the right scale during pre-training. For two policies $\pi$ and $\mu$ , $\pi$ is calibrated with respect to the reference policy $\mu$ if $\mathbb{E}_{a \sim \pi} \left[ Q_{\theta}^{\pi}(s, a) \right] \geq \mathbb{E}_{a \sim \mu} \left[ Q^{\mu}(s, a) \right] := V^{\mu}(s), \, \forall s \in D$ .
  4. Theory and experiments show that a not so good reference policy can have a good calibration effect (because calibration only takes care of the scale of $\hat{Q}$ ). The behavior policy of the large dataset is a good reference, because $Q^{\mu}(s, a)$ can come from crafting reward signal from the success or not signal of the pre-training dataset.
  5. Simple change of algo!! Good theory with good experiment results!

### Flow Matching for Generative Model

> Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., & Le, M. (2022). Flow matching for generative modeling. _arXiv preprint arXiv:2210.02747_.

- Main Question

  - How to gradually transfer a known distribution into another distribution when we only have samples from the latter?
- Main Idea

  - Define a flow $\phi_t{(x)}$ , where $\phi_o(x)=x$ . Define a time variant vector field $\frac{d}{dt}\phi_t(x)$ . Then, starting from point $x$ sampled from a known distribution, if we follow the vector field (the dynamic of the flow), we can **gradually **move into a new distribution.
