//! Semantic-Native Type Systems & Differentiable Interfaces
//!
//! Traditional type systems enforce rigid discrete syntax (`int`, `str`, `Tensor[f32]`).
//! In contrast, next-gen AI requires **continuous semantic typing** governed by latent embedding spaces.
//!
//! Features:
//! - **Semantic Manifold Types (`SemanticConcept`)**: Defines valid types as continuous hyperspheres in $\mathbb{R}^D$
//! - **Continuous Type Checking (`SemanticContract`)**: Returns smooth compatibility scores $\in [0, 1]$
//! - **Differentiable Type Violations**: When output embeddings violate semantic contracts,
//!   emits smooth gradient penalties allowing end-to-end self-correction via gradient descent

use anyhow::{bail, Result};

/// A semantic type defined by an anchor manifold in embedding space.
#[derive(Clone, Debug)]
pub struct SemanticConcept {
    pub name: String,
    /// Center anchor embedding in $\mathbb{R}^D$ representing the conceptual centroid
    pub anchor: Vec<f32>,
    /// Permitted angular or Euclidean tolerance radius
    pub tolerance_radius: f32,
}

impl SemanticConcept {
    pub fn new(name: &str, anchor: Vec<f32>, tolerance_radius: f32) -> Self {
        let norm = l2_norm(&anchor);
        let normalized_anchor = if norm > 1e-12 {
            anchor.iter().map(|&x| x / norm).collect()
        } else {
            anchor
        };
        Self {
            name: name.to_string(),
            anchor: normalized_anchor,
            tolerance_radius,
        }
    }

    pub fn dimension(&self) -> usize {
        self.anchor.len()
    }
}

/// Verification result of a continuous semantic type check.
#[derive(Clone, Debug)]
pub struct TypeCheckResult {
    pub is_valid: bool,
    pub cosine_similarity: f32,
    pub euclidean_distance: f32,
    pub differentiable_penalty: f32,
}

/// Differentiable semantic type contract.
#[derive(Clone, Debug)]
pub struct SemanticContract {
    pub concept: SemanticConcept,
    pub strict_rejection: bool,
}

impl SemanticContract {
    pub fn new(concept: SemanticConcept, strict_rejection: bool) -> Self {
        Self {
            concept,
            strict_rejection,
        }
    }

    /// Evaluates whether an output embedding satisfies the semantic type contract.
    /// Emits a differentiable penalty: $\mathcal{L}_{\text{type}} = \max(0, \text{dist} - \tau)^2$.
    pub fn verify(&self, embedding: &[f32]) -> Result<TypeCheckResult> {
        if embedding.len() != self.concept.dimension() {
            bail!(
                "Embedding dimension {} does not match semantic concept '{}' dimension {}",
                embedding.len(),
                self.concept.name,
                self.concept.dimension()
            );
        }

        let e_norm = l2_norm(embedding);
        let norm_e: Vec<f32> = if e_norm > 1e-12 {
            embedding.iter().map(|&x| x / e_norm).collect()
        } else {
            embedding.to_vec()
        };

        // Cosine similarity
        let cosine_sim: f32 = norm_e
            .iter()
            .zip(self.concept.anchor.iter())
            .map(|(&a, &b)| a * b)
            .sum();

        // Euclidean distance in normalized hypersphere: dist = sqrt(2 - 2*cos)
        let euclidean_dist = (2.0 - 2.0 * cosine_sim.clamp(-1.0, 1.0)).max(0.0).sqrt();

        let violation = euclidean_dist - self.concept.tolerance_radius;
        let differentiable_penalty = if violation > 0.0 {
            violation * violation
        } else {
            0.0
        };

        let is_valid = violation <= 0.0;

        if self.strict_rejection && !is_valid {
            bail!(
                "SemanticTypeError: output embedding fell outside acceptance manifold for concept '{}' (dist: {:.4}, max allowed: {:.4}, cos_sim: {:.4})",
                self.concept.name,
                euclidean_dist,
                self.concept.tolerance_radius,
                cosine_sim
            );
        }

        Ok(TypeCheckResult {
            is_valid,
            cosine_similarity: cosine_sim,
            euclidean_distance: euclidean_dist,
            differentiable_penalty,
        })
    }

    /// Computes analytical gradients of the semantic type penalty with respect to the input embedding:
    /// $$\nabla_e \mathcal{L}_{\text{type}}$$
    pub fn penalty_gradient(&self, embedding: &[f32]) -> Result<Vec<f32>> {
        let check = self.verify(embedding)?;
        let d = self.concept.dimension();

        if check.is_valid || check.euclidean_distance < 1e-8 {
            return Ok(vec![0.0f32; d]);
        }

        let e_norm = l2_norm(embedding);
        let scale = 2.0 * (check.euclidean_distance - self.concept.tolerance_radius) / (check.euclidean_distance * e_norm.max(1e-8));

        // Grad direction points away from concept anchor back towards compliant manifold
        let mut grad = Vec::with_capacity(d);
        for i in 0..d {
            let diff = embedding[i] / e_norm.max(1e-8) - self.concept.anchor[i];
            grad.push(scale * diff);
        }

        Ok(grad)
    }
}

fn l2_norm(slice: &[f32]) -> f32 {
    slice.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_type_compliance() {
        let anchor = vec![1.0, 0.0, 0.0];
        let concept = SemanticConcept::new("MathReasoning", anchor, 0.5);
        let contract = SemanticContract::new(concept, false);

        // Near-identical vector
        let candidate_ok = vec![0.99, 0.05, 0.0];
        let res_ok = contract.verify(&candidate_ok).unwrap();
        assert!(res_ok.is_valid);
        assert_eq!(res_ok.differentiable_penalty, 0.0);

        // Far vector (orthogonal)
        let candidate_bad = vec![0.0, 1.0, 0.0];
        let res_bad = contract.verify(&candidate_bad).unwrap();
        assert!(!res_bad.is_valid);
        assert!(res_bad.differentiable_penalty > 0.0);
    }

    #[test]
    fn test_differentiable_type_gradient_guidance() {
        let anchor = vec![1.0, 0.0];
        let concept = SemanticConcept::new("SafeOutput", anchor, 0.2);
        let contract = SemanticContract::new(concept, false);

        // Disobedient embedding
        let non_compliant = vec![0.0, 1.0];
        let grad = contract.penalty_gradient(&non_compliant).unwrap();

        // Taking a gradient descent step: x_new = x - lr * grad
        let lr = 0.5;
        let corrected = vec![
            non_compliant[0] - lr * grad[0],
            non_compliant[1] - lr * grad[1],
        ];

        let before_dist = contract.verify(&non_compliant).unwrap().euclidean_distance;
        let after_dist = contract.verify(&corrected).unwrap().euclidean_distance;

        // Gradient descent step MUST pull embedding closer to semantic manifold
        assert!(
            after_dist < before_dist,
            "Type penalty gradient must guide embedding closer to target concept manifold"
        );
    }
}
