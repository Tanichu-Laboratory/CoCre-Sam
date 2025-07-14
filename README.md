# CoCre-Sam: Modeling Ouija Board as Collective Langevin Dynamics Sampling from Fused Language Models

This is the official implementation for the paper:  
**CoCre-Sam (Kokkuri-san): Modeling Ouija Board as Collective Langevin Dynamics Sampling from Fused Language Models**

**Authors:** Tadahiro Taniguchi, Masatoshi Nagano, Haruumi Omoto, Yoshiki Hayashi  
**Affiliations:** Kyoto University, Ritsumeikan University, University of Reading

## 🧠 Abstract

Collective human activities like using an Ouija board often produce emergent, coherent linguistic outputs unintended by any single participant. While psychological explanations such as the ideomotor effect exist, a computational understanding of how decentralized, implicit linguistic knowledge fuses through shared physical interaction remains elusive.

This research introduces **CoCre-Sam (Collective-Creature Sampling)**, a framework that models this phenomenon as collective Langevin dynamics sampling from implicitly fused language models. Each participant is represented as an agent with an energy landscape derived from an internal language model.

We theoretically prove that the collective motion of the shared pointer (planchette) corresponds to Langevin MCMC sampling from the sum of individual energy landscapes, representing fused collective knowledge. Simulations validate that CoCre-Sam effectively fuses different models to generate meaningful character sequences, while ablation studies confirm the essential roles of collective interaction and stochasticity.

## 🚀 Running the Simulation

```
pip install -r requirements.txt
```
The main simulation program is located at:

```
src/main.py
```

## ✨ Main Contributions

1. We propose a novel computational interpretation of the Ouija board / Kokkuri-san phenomenon as a collective sampling process from implicitly fused language models, bridging individual cognition and emergent collective behavior.
2. We provide theoretical and empirical justification for the framework:
   - Mathematically grounded in Langevin MCMC.
   - Simulations show effective fusion of language models to generate meaningful sequences.


