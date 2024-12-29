# CaseVPR: Correlation-Aware Sequential Embedding for  Sequence-to-Frame Visual Place Recognition
## Abstract
Visual Place Recognition (VPR) is crucial for autonomous vehicles, as it enables their identification of previously visited locations. 
Compared with conventional single-frame retrieval, leveraging sequences of frames to depict places has been proven effective in alleviating perceptual aliasing.
However, mainstream sequence retrieval methods encode multiple frames into a single descriptor, relinquishing the capacity of fine-grained frame-to-frame matching.
This limitation hampers the precise positioning of individual frames within the query sequence.
On the other hand, sequence matching methods such as SeqSLAM are capable of frame-to-frame matching, but they rely on global brute-force search and the constant speed assumption, which may result in retrieval failures.
To address the above issues, we propose a sequence-to-frame hierarchical matching pipeline for VPR, named CaseVPR.
It consists of coarse-level sequence retrieval based on sequential descriptor matching to mine potential starting points, followed by fine-grained sequence matching to find frame-to-frame correspondence. 
Particularly, a CaseNet is proposed to encode the correlation-aware features of consecutive frames into hierarchical descriptors for sequence retrieval and matching.
On this basis, an AdaptSeq-V2 searching strategy is proposed to identify frame-level correspondences of the query sequence in candidate regions determined by potential starting points.
To validate our hierarchical pipeline, we evaluate CaseVPR on multiple datasets.
Experiments demonstrate that our CaseVPR outperforms all benchmark methods in terms of average precision, and achieves new State-of-the-art (SOTA) results for sequence-based VPR.
## Supplementary Material
Please refer to  [docs/SupplementaryMaterial.pdf](docs/SupplementaryMaterial.pdf) for more ablation studies and qualatative results.
