# QTL for Dementia Detection: Optimization of Feature Maps in MRI

**Student:** Kauê Miziara  
**Advisor:** Dr. Felipe Mahlow

## Introduction

The increase in global life expectancy has brought with it a growth in the number of dementia cases, with Alzheimer's disease being one of the leading causes of disability among the elderly. Early and accurate diagnoses are fundamental for effective therapeutic interventions, and the use of Magnetic Resonance Imaging (MRI) has been the gold standard in aiding the detection of structural biomarkers. Although classical Deep Learning models have achieved notable successes in this task, the high dimensionality of biomedical data and the heterogeneity of images often impose barriers to generalization and require massive computational resources, evidencing a saturation in the performance of purely classical methods [1].

In this scenario, quantum computing emerges as a new frontier for information processing, offering advantages in manipulating states in exponentially dimensional Hilbert spaces. Studies in _Quantum Machine Learning_ (QML) seek to explore properties such as superposition and entanglement to identify patterns that are computationally intractable for classical algorithms [12, 5]. As highlighted in recent literature, the application of these techniques in the diagnosis of neurodegenerative diseases is one of the pillars for the progress of civilization in the current computing era [13]. In particular, the _Quantum Transfer Learning_ (QTL) paradigm allows variational quantum models to be coupled with pre-trained classical neural networks, leveraging high-level feature extraction and refining the final classification through parameterized quantum circuits [2, 4].

Despite the potential, one of the most critical challenges for practical implementation on current noisy devices (_Noisy Intermediate-Scale Quantum_, NISQ) is the _Data Encoding_ or _Feature Mapping_ stage. This process, which performs the projection of classical data into quantum states, defines not only the learning capacity of the model but also its resilience against the intrinsic noise of current hardware [7, 10]. Studies indicate that the choice among encoding techniques (such as basis, amplitude, or rotation) drastically alters the decision boundary [8, 11]. Furthermore, recent research suggests that the optimization of how data is delivered to the circuit (ordering and selection of features) can be just as determining for performance as the _Ansatz_ architecture itself [6].

The state of the art already demonstrates the viability of using QTL for dementia detection using the OASIS-II dataset [2]. However, there remains a need for rigorous benchmarks that compare the theoretical performance in ideal simulators with execution on real Quantum Processing Units (QPUs), ensuring that proposals of quantum advantage hold up against established classical metrics [3].

This research is justified by the gap in investigating how different entanglement topologies and encoding optimization techniques, such as the $ZZ$-_Feature Map_, behave in this specific medical domain, aiming at the development of more effective and noise-resilient computer-assisted diagnoses.

## Objectives

### General Objectives

Investigate the effectiveness of different data _encoding_ strategies (_feature maps_) on the performance and robustness of hybrid QTL models applied to the classification of dementia stages in MRI images, aiming to optimize class separability on NISQ quantum hardware.

### Specific Objectives

- **Baseline Reproduction:** Implement and validate the quantum transfer learning model proposed by Bhowmik et al. [2] using the OASIS-II dataset [9], establishing baseline metrics for accuracy and loss.
- **Feature Maps Implementation:** Develop and integrate varied _encoding_ circuits into the model, including _Angle Encoding_, $ZZ$-_Feature Map_, and other hardware-efficient entanglement topologies, based on the architectures proposed by Suda Neto et al. [13] and Combarro & González-Castillo [4].
- **Encoding Optimization:** Apply classical preprocessing techniques to optimize how image features are delivered to the quantum _Ansatz_ (weighting and ordering of _features_), as suggested by Fioravanti et al. [6].
- **Benchmarking on Real Hardware:** Perform a comparative performance analysis between classical state-vector simulations and executions on real IBM QPUs, evaluating the impact of circuit depth on the error rate.
- **Noise Resilience Analysis:** Quantitatively evaluate the resilience of each _encoding_ strategy against different noise models (such as depolarization and thermal relaxation), using the decision boundary stability metrics discussed by LaRose & Coyle [8] and Munikote et al. [11].
- **Scientific Production:** Document the obtained results in a technical article, detailing the practical viability of QML techniques for the computer-assisted diagnosis of neurodegenerative diseases.

## Methodology

The execution of this project is structured into four fundamental technical phases, integrating biomedical data processing, variational modeling, and experimentation on real hardware.

### Data Preparation and Preprocessing

The study will utilize the OASIS-II dataset (_Open Access Series of Imaging Studies_, [9]), consisting of longitudinal Magnetic Resonance images. Since the collection and cleaning phases have already been completed, the methodology focuses on integrating the NIfTI and HDR format files into an optimized ETL (_Extract, Transform, Load_) pipeline. The images will undergo _Min-Max_ normalization to suit the rotation limits of quantum gates, ensuring that pixel values are mapped in the interval $[0, \pi]$ or $[0, 2\pi]$ [13, 11].

### Hybrid Architecture Development (Dressed Quantum Network)

The research adopts the _Quantum Transfer Learning_ (QTL) paradigm. The classical feature extraction model (e.g., _ResNet_ or similar pre-trained architectures) will serve as a backbone, whose high-dimensional outputs will be reduced via linear layers to feed the Variational Quantum Circuit (VQC).

The development will be cross-platform:

- **Python (PennyLane and Qiskit):** For integration with deep neural networks (_PyTorch_) and for communication with IBM QPUs via the cloud.
- **Julia (Yao.jl):** Used for large-scale circuit and state-vector simulations that require high computational performance. The extensibility of _Yao.jl_ will allow for the rapid prototyping of new _Ansätze_ and the analysis of quantum automatic differentiation with less computational overhead than traditional frameworks [14].

### Feature Maps Experimentation and Optimization

The expansion phase will explore the effectiveness of different _encodings_. In addition to the _Angle Embedding_ used in the baseline [2], the $ZZ$-_Feature Map_ and higher-order encodings will be implemented. Following the propositions of Fioravanti et al. [6], a pre-circuit classical optimization step will be applied to determine the ordering and weighting of the most relevant _features_, aiming to maximize class separability in Hilbert space before applying the variational _Ansatz_.

### Benchmarking, Transpilation, and Execution on Hardware

The models will be validated in two distinct environments. First, in ideal simulators (state-vector) to establish the theoretical accuracy limit. Subsequently, the circuits will be submitted to real QPUs. Due to the student's position as a _Qiskit Advocate_, free execution on selected IBM QPUs for up to 600 seconds is guaranteed, renewing monthly. It is also possible to extend the plan to 180 free minutes that can be used within a year, with no monthly limit.

In this stage, advanced transpilation techniques (optimization levels 2 and 3 in Qiskit) will be applied to reduce circuit depth and the number of $CNOT$ gates, mitigating the effects of decoherence and readout errors typical of the NISQ era [7, 8]. The results will be compared through accuracy metrics, confusion matrix, and noise resilience analysis.

## Execution Schedule

The project is structured to be executed over a 12-month period, with an expected start in May 2026. Since the preliminary data collection and cleaning stage for the OASIS-II dataset has already been completed, the schedule starts directly in the integration, modeling, and experimentation phases.

The activities have been divided into the following macro-stages, also described in the calendar presented in Table 1:

- **A1. Baseline Integration and Reproduction:** Loading data into the ETL pipeline, implementing the classical-quantum hybrid architecture using _Angle Embedding_, and executing the first training sessions in classical simulators to establish reference metrics.
- **A2. Alternative Feature Maps Modeling:** Development and mathematical implementation of new encoding circuits, including the $ZZ$-_Feature Map_ and custom entanglement topologies, using Python (Qiskit/PennyLane) and Julia (Yao.jl) for state-vector simulations.
- **A3. Classical Encoding Optimization:** Application of classical preprocessing algorithms to optimize the ordering and weighting of the _features_ that will feed the quantum circuit, aiming to maximize class separability.
- **A4. Transpilation and Hardware Execution (QPUs):** Adaptation of the circuits to the physical constraints of IBM hardware, application of optimization levels in the transpiler, and submission of the variational models to processing queues on real quantum computers, utilizing the free plan.
- **A5. Evaluation, Noise Analysis, and Benchmarking:** Collection of empirical data, application of simulated noise models (depolarization/relaxation), and statistical comparison among the baseline, simulated quantum models, and those executed on real hardware.
- **A6. Scientific Production and Final Report:** Drafting and review of the final report, in addition to compiling the results into a scientific article format for submission to conferences or specialized journals.

### Table 1: Execution Schedule (2026)

| Months                 | A1  | A2  | A3  | A4  | A5  | A6  |
| :--------------------- | :-: | :-: | :-: | :-: | :-: | :-: |
| **Month 1 and 2**      |  X  |     |     |     |     |     |
| **Month 3**            |  X  |  X  |     |     |     |     |
| **Month 4 and 5**      |     |  X  |  X  |     |     |     |
| **Month 6**            |     |  X  |  X  |  X  |     |  X  |
| **Month 7 and 8**      |     |     |  X  |  X  |  X  |  X  |
| **Month 9, 10 and 11** |     |     |     |  X  |  X  |  X  |
| **Month 12**           |     |     |     |     |  X  |  X  |

## References

[1] Akpinar, E., & Oduncuoglu, M. (2026). _Quantum Model Parallelism for MRI-Based Classification of Alzheimer's Disease Stages_. arXiv preprint arXiv:2602.00128.
[2] Bhowmik, S., Perciano, T., & Thapliyal, H. (2025). _Quantum Transfer Learning to Boost Dementia Detection_. Proceedings of the Great Lakes Symposium on VLSI 2025, 849–853.
[3] Chang, S. Y., & Cerezo, M. (2025). _A Primer on Quantum Machine Learning_. arXiv preprint arXiv:2511.15969.
[4] Combarro, E. F., & González-Castillo, S. (2023). _A Practical Guide to Quantum Machine Learning and Quantum Optimization: Hands-on Approach to Modern Quantum Algorithms_. Packt Publishing.
[5] Du, Y., Wang, X., Guo, N., Yu, Z., Qian, Y., Zhang, K., Hsieh, M.-H., Rebentrost, P., & Tao, D. (2025). _Quantum Machine Learning: A Hands-on Tutorial for Machine Learning Practitioners and Researchers_. arXiv preprint arXiv:2502.01146.
[6] Fioravanti, T., Quanz, B., Agliardi, G., Guzman, E. A. R., Carrascal, G., & Park, J.-E. (2025). _Quantum feature encoding optimization_. arXiv preprint arXiv:2512.02422.
[7] Gujju, Y., Matsuo, A., & Raymond, R. (2024). _Quantum machine learning on near-term quantum devices: Current state of supervised and unsupervised techniques for real-world applications_. Physical Review Applied, 21(6), 067001.
[8] LaRose, R., & Coyle, B. (2020). _Robust data encodings for quantum classifiers_. Physical Review A, 102(3), 032420.
[9] Marcus, D. L., Wang, J., Parker, J., Csernansky, J. G., Morris, J. C., & Buckner, R. L. (2007). _Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI Data in Young, Middle Aged, Nondemented and Demented Older Adults_. Journal of Computer Assisted Tomography, 31(6), 1498–1504.
[10] Micklethwaite, E., & Lowe, A. (2025). _Classification using quantum kernels in a radial basis function network_. arXiv preprint arXiv:2512.20567.
[11] Munikote, N., Li, A., Liu, C., & Stein, S. (2024). _Comparing Quantum Encoding Techniques_. arXiv preprint arXiv:2410.09121.
[12] Peral-García, D., Cruz-Benito, J., & García-Peñalvo, F. J. (2024). _Systematic literature review: Quantum machine learning and its applications_. Computer Science Review, 51, 100619.
[13] Suda Neto, J., Fanchini, F. F., de Oliveira, M. C., Arruda, L. G. E., & Guido, R. C. (2025). _Aprendizado de Máquina Quântica: Teoria e Aplicações_. Independente.
[14] Luo, X.-Z., Liu, J.-G., Zhang, P., & Wang, L. (2020). _Yao.jl: Extensible, Efficient Framework for Quantum Algorithm Design_. Quantum, 4, 341.
