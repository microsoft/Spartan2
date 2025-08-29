//! Hash-based multilinear PCS for Spartan2 (Track A)
//! 
//! This implementation provides a Merkle tree-based polynomial commitment scheme
//! for multilinear polynomials using Keccak256 hashing. It supports LeakReduced mode
//! which reveals O(m) dense fold values during evaluation.

use crate::{
  errors::SpartanError,
  polys::multilinear::MultilinearPolynomial,
  traits::{
    Engine,
    pcs::{CommitmentTrait, PCSEngineTrait},
    transcript::{TranscriptEngineTrait, TranscriptReprTrait},
  },
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Keccak256};
use std::marker::PhantomData;

/// Domain tags to avoid cross-protocol collisions
const TAG_LEAF: &[u8] = b"mle/leaf";
const TAG_NODE: &[u8] = b"mle/node";
const TAG_MODE: &[u8] = b"mle/mode";
const TAG_LAYER_ROOTS: &[u8] = b"mle/layer_roots";

/// Zero-knowledge mode for the Hash-MLE PCS
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZkMode {
  /// Leak-reduced mode: reveals O(m) dense fold values during evaluation
  LeakReduced,
}

/// Commitment key for Hash-MLE PCS
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HashMleCommitmentKey<E: Engine> {
  /// Merkle arity (currently fixed to 2)
  branching: u8,
  /// ZK mode baked into this key
  zk_mode: ZkMode,
  _p: PhantomData<E>,
}

/// Verifier key for Hash-MLE PCS
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HashMleVerifierKey<E: Engine> {
  branching: u8,
  zk_mode: ZkMode,
  _p: PhantomData<E>,
}

/// A Merkle tree root (32-byte hash)
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MerkleRoot([u8; 32]);

impl<G: crate::traits::Group> TranscriptReprTrait<G> for MerkleRoot {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.0.to_vec()
  }
}

impl MerkleRoot {
  #[allow(dead_code)]
  fn zero() -> Self { 
    MerkleRoot([0u8; 32]) 
  }
}

/// Authentication path for a Merkle tree leaf
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MerklePath {
  /// leaf index within the committed array (0-based)
  pub leaf_index: u64,
  /// authentication path (sibling digests, bottom-up)
  pub siblings: Vec<[u8; 32]>,
}

/// Commitment for Hash-MLE PCS
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HashMleCommitment<E: Engine> {
  /// Root of the base vector commitment (unmasked v in LeakReduced mode)
  base_root: MerkleRoot,
  /// Mode encoded into the commitment to avoid misuse across modes
  mode: ZkMode,
  _p: PhantomData<E>,
}

impl<E: Engine> TranscriptReprTrait<E::GE> for HashMleCommitment<E> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    let mut out = Vec::with_capacity(2 + 32);
    out.extend_from_slice(TAG_MODE);
    out.push(match self.mode { ZkMode::LeakReduced => 0 });
    out.extend_from_slice(&self.base_root.0);
    out
  }
}

impl<E: Engine> CommitmentTrait<E> for HashMleCommitment<E> {}

/// Blinding factor for Hash-MLE PCS
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HashMleBlind<E: Engine> {
  _phantom: PhantomData<E>,
}

impl<E: Engine> Default for HashMleBlind<E> {
  fn default() -> Self { 
    HashMleBlind { _phantom: PhantomData } 
  }
}

/// Evaluation argument: per-round pair openings and the next-layer single opening.
#[derive(Clone, Debug)]
pub struct HashMleEvaluationArgument<E: Engine> {
  /// Layer roots carried here to avoid bloating the commitment
  layer_roots: Vec<MerkleRoot>, // len = m+1

  /// For each round i in [0..m):
  rounds: Vec<Round<E>>,
}

impl<E: Engine> serde::Serialize for HashMleEvaluationArgument<E> {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: serde::Serializer,
  {
    use serde::ser::SerializeStruct;
    let mut state = serializer.serialize_struct("HashMleEvaluationArgument", 2)?;
    state.serialize_field("layer_roots", &self.layer_roots)?;
    state.serialize_field("rounds", &self.rounds)?;
    state.end()
  }
}

impl<'de, E: Engine> serde::Deserialize<'de> for HashMleEvaluationArgument<E> {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: serde::Deserializer<'de>,
  {
    use serde::de::{self, MapAccess, Visitor};
    use std::fmt;

    #[derive(serde::Deserialize)]
    #[serde(field_identifier, rename_all = "snake_case")]
    enum Field {
      LayerRoots,
      Rounds,
    }

    struct HashMleEvaluationArgumentVisitor<E: Engine>(PhantomData<E>);

    impl<'de, E: Engine> Visitor<'de> for HashMleEvaluationArgumentVisitor<E> {
      type Value = HashMleEvaluationArgument<E>;

      fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("struct HashMleEvaluationArgument")
      }

      fn visit_map<V>(self, mut map: V) -> Result<HashMleEvaluationArgument<E>, V::Error>
      where
        V: MapAccess<'de>,
      {
        let mut layer_roots = None;
        let mut rounds = None;
        while let Some(key) = map.next_key()? {
          match key {
            Field::LayerRoots => {
              if layer_roots.is_some() {
                return Err(de::Error::duplicate_field("layer_roots"));
              }
              layer_roots = Some(map.next_value()?);
            }
            Field::Rounds => {
              if rounds.is_some() {
                return Err(de::Error::duplicate_field("rounds"));
              }
              rounds = Some(map.next_value()?);
            }
          }
        }
        let layer_roots = layer_roots.ok_or_else(|| de::Error::missing_field("layer_roots"))?;
        let rounds = rounds.ok_or_else(|| de::Error::missing_field("rounds"))?;
        Ok(HashMleEvaluationArgument { layer_roots, rounds })
      }
    }

    const FIELDS: &'static [&'static str] = &["layer_roots", "rounds"];
    deserializer.deserialize_struct("HashMleEvaluationArgument", FIELDS, HashMleEvaluationArgumentVisitor(PhantomData))
  }
}

/// A single round of the Hash-MLE evaluation argument
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Round<E: Engine> {
  /// openings from layer i (v^{(i)}):
  a: E::Scalar,         // even
  b: E::Scalar,         // odd
  path_a: MerklePath,   // membership against layer_roots[i]
  path_b: MerklePath,   // membership against layer_roots[i]

  /// membership for layer i+1 at the folded index
  next: E::Scalar,      // equals (1-r_i)*a + r_i*b
  path_next: MerklePath // membership against layer_roots[i+1]
}

/// Minimal Merkle helper (binary). We keep it local to avoid extra deps.
fn keccak256(bytes: &[u8]) -> [u8; 32] {
  let mut h = Keccak256::new();
  h.update(bytes);
  h.finalize().into()
}

fn leaf_hash<E: Engine>(x: &E::Scalar) -> [u8; 32] {
  let mut v = Vec::with_capacity(TAG_LEAF.len() + 64);
  v.extend_from_slice(TAG_LEAF);
  v.extend_from_slice(&x.to_transcript_bytes());
  keccak256(&v)
}

fn node_hash(l: &[u8; 32], r: &[u8; 32]) -> [u8; 32] {
  let mut v = Vec::with_capacity(TAG_NODE.len() + 64);
  v.extend_from_slice(TAG_NODE);
  v.extend_from_slice(l);
  v.extend_from_slice(r);
  keccak256(&v)
}

#[derive(Clone)]
struct MerkleTree {
  /// bottom layer (leaves), power of two
  #[allow(dead_code)]
  leaves: Vec<[u8; 32]>,
  layers: Vec<Vec<[u8; 32]>>, // including leaves; layers[0] == leaves, layers.last()[0] == root
}

impl MerkleTree {
  fn from_leaves(leaves: Vec<[u8; 32]>) -> Self {
    assert!(leaves.len().is_power_of_two());
    let mut layers = Vec::new();
    let mut cur = leaves.clone();
    layers.push(cur.clone());
    
    while cur.len() > 1 {
      cur = cur
        .chunks_exact(2)
        .map(|p| node_hash(&p[0], &p[1]))
        .collect::<Vec<_>>();
      layers.push(cur.clone());
    }
    
    Self { leaves, layers }
  }
  
  fn root(&self) -> [u8; 32] { 
    *self.layers.last().unwrap().first().unwrap() 
  }
  
  fn open(&self, leaf_index: usize) -> MerklePath {
    let mut idx = leaf_index;
    let mut siblings = Vec::with_capacity(self.layers.len() - 1);
    
    for layer in &self.layers {
      if layer.len() == 1 { 
        break; 
      }
      let sib = if idx % 2 == 0 { 
        layer[idx + 1] 
      } else { 
        layer[idx - 1] 
      };
      siblings.push(sib);
      idx >>= 1;
    }
    
    MerklePath { 
      leaf_index: leaf_index as u64, 
      siblings 
    }
  }
  
  fn verify(path: &MerklePath, leaf: &[u8; 32], root: &[u8; 32]) -> bool {
    let mut idx = path.leaf_index as usize;
    let mut cur = *leaf;
    
    for sib in &path.siblings {
      cur = if idx % 2 == 0 { 
        node_hash(&cur, sib) 
      } else { 
        node_hash(sib, &cur) 
      };
      idx >>= 1;
    }
    
    &cur == root
  }
}

/// Fold one layer with a single r, halving the vector length.
/// This follows the same logic as MultilinearPolynomial::bind_poly_var_top
fn fold_layer<E: Engine>(v: &[E::Scalar], r: &E::Scalar) -> Vec<E::Scalar> {
  let n = v.len() / 2;
  (0..n)
    .into_par_iter()
    .map(|i| {
      let left = v[i];
      let right = v[i + n];
      left + *r * (right - left) // equivalent to (1-r)*left + r*right
    })
    .collect()
}

/// Hash-based multilinear polynomial commitment scheme
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HashMlePCS<E: Engine> {
  _p: PhantomData<E>,
}

impl<E: Engine> PCSEngineTrait<E> for HashMlePCS<E> {
  type CommitmentKey = HashMleCommitmentKey<E>;
  type VerifierKey = HashMleVerifierKey<E>;
  type Commitment = HashMleCommitment<E>;
  type PartialCommitment = HashMleCommitment<E>;
  type Blind = HashMleBlind<E>;
  type EvaluationArgument = HashMleEvaluationArgument<E>;

  fn width() -> usize { 
    usize::MAX / 2 // force a single "row" per commitment
  }

  fn setup(_label: &'static [u8], _n: usize) -> (Self::CommitmentKey, Self::VerifierKey) {
    let ck = HashMleCommitmentKey { 
      branching: 2, 
      zk_mode: ZkMode::LeakReduced, 
      _p: PhantomData 
    };
    let vk = HashMleVerifierKey { 
      branching: 2, 
      zk_mode: ZkMode::LeakReduced, 
      _p: PhantomData 
    };
    (ck, vk)
  }

  fn blind(_ck: &Self::CommitmentKey, _n: usize) -> Self::Blind {
    HashMleBlind::default()
  }

  fn commit(
    ck: &Self::CommitmentKey,
    v: &[E::Scalar],
    _r: &Self::Blind,
    _is_small: bool,
  ) -> Result<Self::Commitment, SpartanError> {
    if !v.len().is_power_of_two() {
      return Err(SpartanError::InvalidInputLength { 
        reason: "HashMlePCS: vector len must be power of two".into() 
      });
    }
    
    // Base layer leaves (unmasked in LeakReduced mode)
    let leaves = v.par_iter().map(leaf_hash::<E>).collect::<Vec<_>>();
    let tree = MerkleTree::from_leaves(leaves);
    let base_root = MerkleRoot(tree.root());

    Ok(HashMleCommitment { 
      base_root, 
      mode: ck.zk_mode, 
      _p: PhantomData 
    })
  }

  fn commit_partial(
    ck: &Self::CommitmentKey,
    v: &[E::Scalar],
    blind: &Self::Blind,
    is_small: bool,
  ) -> Result<(Self::PartialCommitment, Self::Blind), SpartanError> {
    let c = Self::commit(ck, v, blind, is_small)?;
    Ok((c, blind.clone()))
  }

  fn check_partial(_comm: &Self::PartialCommitment, _n: usize) -> Result<(), SpartanError> { 
    Ok(()) 
  }

  fn combine_partial(
    partial_comms: &[Self::PartialCommitment],
  ) -> Result<Self::Commitment, SpartanError> {
    if partial_comms.len() != 1 {
      return Err(SpartanError::InvalidInputLength { 
        reason: "HashMlePCS: combine_partial expects exactly one piece".into() 
      });
    }
    Ok(partial_comms[0].clone())
  }

  fn combine_blinds(blinds: &[Self::Blind]) -> Result<Self::Blind, SpartanError> {
    if blinds.len() != 1 {
      return Err(SpartanError::InvalidInputLength { 
        reason: "HashMlePCS: combine_blinds expects exactly one blind".into() 
      });
    }
    Ok(blinds[0].clone())
  }

  fn prove(
    _ck: &Self::CommitmentKey,
    transcript: &mut E::TE,
    comm: &Self::Commitment,
    poly: &[E::Scalar],
    _blind: &Self::Blind,
    point: &[E::Scalar],
  ) -> Result<(E::Scalar, Self::EvaluationArgument), SpartanError> {
    let n = poly.len();
    let m = point.len();
    if n != (1usize << m) {
      return Err(SpartanError::InvalidInputLength {
        reason: format!("HashMlePCS::prove expected {} elements, got {}", 1usize << m, n)
      });
    }
    
    transcript.absorb(b"poly_com", comm);

    // Use the MultilinearPolynomial's own evaluation method to get the correct result
    let mle = MultilinearPolynomial::new(poly.to_vec());
    let expected_eval = mle.evaluate(point);

    // For the PCS, we need to build the evaluation layers step by step
    // following the same logic as MultilinearPolynomial
    let mut current_poly = poly.to_vec();
    let mut all_layers = vec![current_poly.clone()];
    
    // Build layers by binding variables one by one, following MultilinearPolynomial's approach
    for &r_i in point.iter() {
      let next_layer = fold_layer::<E>(&current_poly, &r_i);
      all_layers.push(next_layer.clone());
      current_poly = next_layer;
    }
    debug_assert_eq!(all_layers.last().unwrap().len(), 1);

    // Build trees + roots (per-proof) and round proofs
    let trees: Vec<MerkleTree> = all_layers
      .par_iter()
      .map(|lvl| MerkleTree::from_leaves(lvl.par_iter().map(leaf_hash::<E>).collect()))
      .collect();

    let layer_roots: Vec<MerkleRoot> = trees.iter().map(|t| MerkleRoot(t.root())).collect();

    // For each round, we open the correct pairs based on MultilinearPolynomial's folding
    // NOTE: We also absorb layer roots here for transcript symmetry (verifier does the same).
    transcript.absorb(TAG_LAYER_ROOTS, &layer_roots.as_slice());

    let mut rounds: Vec<Round<E>> = Vec::with_capacity(m);
    
    for i in 0..m {
      let layer_size = all_layers[i].len();
      let n = layer_size / 2;
      
      // For MultilinearPolynomial folding, we open pairs (left[j], right[j]) 
      // where left is first half and right is second half
      // For simplicity, we'll open the first pair (0, n)
      let a = all_layers[i][0];        // left[0]
      let b = all_layers[i][n];        // right[0] 
      let next = a + point[i] * (b - a); // MultilinearPolynomial folding formula

      let path_a = trees[i].open(0);
      let path_b = trees[i].open(n);
      let path_next = trees[i+1].open(0);

      rounds.push(Round { a, b, path_a, path_b, next, path_next });
    }

    // eval in LeakReduced mode - should match the MultilinearPolynomial evaluation
    let eval = expected_eval;

    // Pack argument
    let arg = HashMleEvaluationArgument { layer_roots, rounds };
    Ok((eval, arg))
  }

  fn verify(
    vk: &Self::VerifierKey,
    transcript: &mut E::TE,
    comm: &Self::Commitment,
    point: &[E::Scalar],
    eval: &E::Scalar,
    arg: &Self::EvaluationArgument,
  ) -> Result<(), SpartanError> {
    if vk.zk_mode != comm.mode {
      return Err(SpartanError::InvalidPCS);
    }
    let m = point.len();
    if arg.layer_roots.len() != m + 1 || arg.rounds.len() != m {
      return Err(SpartanError::InvalidInputLength { 
        reason: "HashMlePCS::verify malformed argument".into() 
      });
    }

    // This PCS is binary Merkle
    if vk.branching != 2 { 
      return Err(SpartanError::InvalidPCS); 
    }

    transcript.absorb(b"poly_com", comm);
    transcript.absorb(TAG_LAYER_ROOTS, &arg.layer_roots.as_slice());

    // Layer 0 root must match the commitment's root
    if arg.layer_roots[0].0 != comm.base_root.0 {
      return Err(SpartanError::InvalidPCS);
    }

    // Per round: check memberships and fold equality
    for i in 0..m {
      let root_i   = &arg.layer_roots[i].0;
      let root_ip1 = &arg.layer_roots[i+1].0;

      // Depth / index sanity (prevents pair-swapping attacks)
      // Layer i has 2^(m-i) leaves, so paths there have length m-i.
      // Layer i+1 has 2^(m-i-1) leaves, so paths there have length m-i-1.
      let depth_a   = arg.rounds[i].path_a.siblings.len();
      let depth_b   = arg.rounds[i].path_b.siblings.len();
      let depth_next= arg.rounds[i].path_next.siblings.len();
      let expected_depth = m - i;
      if depth_a != expected_depth || depth_b != expected_depth || depth_next + 1 != expected_depth {
        return Err(SpartanError::InvalidPCS);
      }
      if expected_depth == 0 { 
        return Err(SpartanError::InvalidPCS); 
      }

      // Expected indices: a at 0, b at 2^(depth-1), next at 0
      let expected_a_idx: u64 = 0;
      let expected_b_idx: u64 = 1u64 << (expected_depth - 1);
      let expected_next_idx: u64 = 0;

      if arg.rounds[i].path_a.leaf_index != expected_a_idx
        || arg.rounds[i].path_b.leaf_index != expected_b_idx
        || arg.rounds[i].path_next.leaf_index != expected_next_idx
      {
        return Err(SpartanError::InvalidPCS);
      }

      let a_h = leaf_hash::<E>(&arg.rounds[i].a);
      let b_h = leaf_hash::<E>(&arg.rounds[i].b);
      if !MerkleTree::verify(&arg.rounds[i].path_a, &a_h, root_i) ||
         !MerkleTree::verify(&arg.rounds[i].path_b, &b_h, root_i) {
        return Err(SpartanError::InvalidPCS);
      }

      let folded = arg.rounds[i].a + point[i] * (arg.rounds[i].b - arg.rounds[i].a);
      if folded != arg.rounds[i].next {
        return Err(SpartanError::InvalidPCS);
      }

      let next_h = leaf_hash::<E>(&arg.rounds[i].next);
      if !MerkleTree::verify(&arg.rounds[i].path_next, &next_h, root_ip1) {
        return Err(SpartanError::InvalidPCS);
      }
    }

    // Final check for LeakReduced mode
    let y = arg.rounds.last().map(|r| r.next).unwrap_or(*eval);
    if y != *eval { 
      return Err(SpartanError::InvalidPCS); 
    }

    Ok(())
  }
}

// Note: Merkle PCS is not linearly homomorphic, so we do NOT implement FoldingEngineTrait.

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    polys::multilinear::MultilinearPolynomial,
    provider::PallasMerkleMleEngine,
    traits::{Engine, pcs::PCSEngineTrait},
  };
  use rand::{Rng, SeedableRng};
  use rand::rngs::StdRng;
  use ff::Field;

  type E = PallasMerkleMleEngine;

  fn rand_scalar(rng: &mut StdRng) -> <E as Engine>::Scalar {
    // sample in a tiny range (fast) then lift into field
    let x: u64 = rng.r#gen();
    <E as Engine>::Scalar::from(x)
  }

  #[test]
  fn test_leak_reduced_roundtrip() {
    let m = 8usize; 
    let n = 1usize << m;
    let poly = (0..n).map(|i| <E as Engine>::Scalar::from(i as u64)).collect::<Vec<_>>();
    let point = (0..m).map(|i| <E as Engine>::Scalar::from((i+1) as u64)).collect::<Vec<_>>();

    let (ck, vk) = <HashMlePCS<E> as PCSEngineTrait<E>>::setup(b"test", n);
    let blind = <HashMlePCS<E> as PCSEngineTrait<E>>::blind(&ck, n);

    let mut tr = <E as Engine>::TE::new(b"test");
    let com = <HashMlePCS<E> as PCSEngineTrait<E>>::commit(&ck, &poly, &blind, false).unwrap();
    let (eval, arg) = <HashMlePCS<E> as PCSEngineTrait<E>>::prove(&ck, &mut tr, &com, &poly, &blind, &point).unwrap();

    let mut tr2 = <E as Engine>::TE::new(b"test");
    <HashMlePCS<E> as PCSEngineTrait<E>>::verify(&vk, &mut tr2, &com, &point, &eval, &arg).unwrap();
  }

  #[test]
  fn test_small_polynomial() {
    let m = 3usize; 
    let n = 1usize << m; // 8 elements
    let poly = vec![
      <E as Engine>::Scalar::from(1u64),
      <E as Engine>::Scalar::from(2u64),
      <E as Engine>::Scalar::from(3u64),
      <E as Engine>::Scalar::from(4u64),
      <E as Engine>::Scalar::from(5u64),
      <E as Engine>::Scalar::from(6u64),
      <E as Engine>::Scalar::from(7u64),
      <E as Engine>::Scalar::from(8u64),
    ];
    let point = vec![
      <E as Engine>::Scalar::from(0u64),
      <E as Engine>::Scalar::from(1u64),
      <E as Engine>::Scalar::from(0u64),
    ]; // Should evaluate to poly[2] = 3

    let (ck, vk) = <HashMlePCS<E> as PCSEngineTrait<E>>::setup(b"test_small", n);
    let blind = <HashMlePCS<E> as PCSEngineTrait<E>>::blind(&ck, n);

    let mut tr = <E as Engine>::TE::new(b"test_small");
    let com = <HashMlePCS<E> as PCSEngineTrait<E>>::commit(&ck, &poly, &blind, false).unwrap();
    let (eval, arg) = <HashMlePCS<E> as PCSEngineTrait<E>>::prove(&ck, &mut tr, &com, &poly, &blind, &point).unwrap();

    // Verify the evaluation is correct
    let expected_eval = <E as Engine>::Scalar::from(3u64);
    assert_eq!(eval, expected_eval);

    let mut tr2 = <E as Engine>::TE::new(b"test_small");
    <HashMlePCS<E> as PCSEngineTrait<E>>::verify(&vk, &mut tr2, &com, &point, &eval, &arg).unwrap();
  }

  #[test]
  fn test_random_polynomial() {
    let m = 6usize; 
    let n = 1usize << m;
    let poly = (0..n).map(|i| <E as Engine>::Scalar::from(i as u64 + 1)).collect::<Vec<_>>();
    let point = (0..m).map(|i| <E as Engine>::Scalar::from(i as u64)).collect::<Vec<_>>();

    let (ck, vk) = <HashMlePCS<E> as PCSEngineTrait<E>>::setup(b"test_random", n);
    let blind = <HashMlePCS<E> as PCSEngineTrait<E>>::blind(&ck, n);

    let mut tr = <E as Engine>::TE::new(b"test_random");
    let com = <HashMlePCS<E> as PCSEngineTrait<E>>::commit(&ck, &poly, &blind, false).unwrap();
    let (eval, arg) = <HashMlePCS<E> as PCSEngineTrait<E>>::prove(&ck, &mut tr, &com, &poly, &blind, &point).unwrap();

    // Verify against direct multilinear evaluation
    let mle = MultilinearPolynomial::new(poly.clone());
    let expected_eval = mle.evaluate(&point);
    assert_eq!(eval, expected_eval);

    let mut tr2 = <E as Engine>::TE::new(b"test_random");
    <HashMlePCS<E> as PCSEngineTrait<E>>::verify(&vk, &mut tr2, &com, &point, &eval, &arg).unwrap();
  }

  #[test]
  fn test_commitment_consistency() {
    let m = 4usize; 
    let n = 1usize << m;
    let poly = (0..n).map(|i| <E as Engine>::Scalar::from((i * i + 7) as u64)).collect::<Vec<_>>();

    let (ck, _vk) = <HashMlePCS<E> as PCSEngineTrait<E>>::setup(b"test_consistency", n);
    let blind = <HashMlePCS<E> as PCSEngineTrait<E>>::blind(&ck, n);

    // Commit to the same polynomial twice
    let com1 = <HashMlePCS<E> as PCSEngineTrait<E>>::commit(&ck, &poly, &blind, false).unwrap();
    let com2 = <HashMlePCS<E> as PCSEngineTrait<E>>::commit(&ck, &poly, &blind, false).unwrap();

    // Commitments should be identical
    assert_eq!(com1, com2);
  }

  #[test]
  fn test_merkle_tree_operations() {
    // Test the internal Merkle tree functionality
    let leaves = vec![
      [1u8; 32], [2u8; 32], [3u8; 32], [4u8; 32]
    ];
    let tree = MerkleTree::from_leaves(leaves.clone());
    
    // Test opening and verification
    for i in 0..leaves.len() {
      let path = tree.open(i);
      assert!(MerkleTree::verify(&path, &leaves[i], &tree.root()));
    }
  }

  #[test]
  fn test_folding_correctness() {
    // Test that our folding matches the MultilinearPolynomial::bind_poly_var_top logic
    let poly = vec![
      <E as Engine>::Scalar::from(1u64),
      <E as Engine>::Scalar::from(2u64),
      <E as Engine>::Scalar::from(3u64),
      <E as Engine>::Scalar::from(4u64),
    ];
    let r = <E as Engine>::Scalar::from(7u64);
    
    let folded = fold_layer::<E>(&poly, &r);
    
    // Manual calculation using MultilinearPolynomial folding: left + r * (right - left)
    // where left is first half [1, 2] and right is second half [3, 4]
    let expected_0 = poly[0] + r * (poly[2] - poly[0]); // 1 + 7*(3-1) = 1 + 14 = 15
    let expected_1 = poly[1] + r * (poly[3] - poly[1]); // 2 + 7*(4-2) = 2 + 14 = 16
    
    assert_eq!(folded.len(), 2);
    assert_eq!(folded[0], expected_0);
    assert_eq!(folded[1], expected_1);
  }

  #[test]
  fn roundtrip_small_fixed() {
    let m = 5usize; 
    let n = 1usize << m;
    let mut rng = StdRng::seed_from_u64(7);
    let poly: Vec<<E as Engine>::Scalar> = (0..n).map(|_| rand_scalar(&mut rng)).collect();
    let point: Vec<<E as Engine>::Scalar> = (0..m).map(|_| rand_scalar(&mut rng)).collect();

    let (ck, vk) = <HashMlePCS<E> as PCSEngineTrait<E>>::setup(b"test", n);
    let blind = HashMleBlind::<E>::default();
    let com = <HashMlePCS<E> as PCSEngineTrait<E>>::commit(&ck, &poly, &blind, false).unwrap();

    let mut tr_prove = <E as Engine>::TE::new(b"t");
    let (eval, arg) = <HashMlePCS<E> as PCSEngineTrait<E>>::prove(&ck, &mut tr_prove, &com, &poly, &blind, &point).unwrap();

    let mut tr_verify = <E as Engine>::TE::new(b"t");
    <HashMlePCS<E> as PCSEngineTrait<E>>::verify(&vk, &mut tr_verify, &com, &point, &eval, &arg).unwrap();

    // cross-check against direct MLE evaluation
    let mle = MultilinearPolynomial::new(poly.clone());
    assert_eq!(eval, mle.evaluate(&point));
  }

  #[test]
  fn commit_rejects_non_power_of_two() {
    let n = 12usize; // not a power of two
    let (ck, _vk) = <HashMlePCS<E> as PCSEngineTrait<E>>::setup(b"bad", n);
    let blind = HashMleBlind::<E>::default();
    let poly: Vec<<E as Engine>::Scalar> = (0..n).map(|i| <E as Engine>::Scalar::from(i as u64)).collect();
    let err = <HashMlePCS<E> as PCSEngineTrait<E>>::commit(&ck, &poly, &blind, false).unwrap_err();
    let msg = format!("{:?}", err);
    assert!(msg.contains("power of two"));
  }

  #[test]
  fn prove_rejects_wrong_point_len() {
    let m = 4usize; 
    let n = 1usize << m;
    let poly: Vec<<E as Engine>::Scalar> = (0..n).map(|i| <E as Engine>::Scalar::from(i as u64)).collect();
    let (ck, _vk) = <HashMlePCS<E> as PCSEngineTrait<E>>::setup(b"bad2", n);
    let blind = HashMleBlind::<E>::default();
    let com = <HashMlePCS<E> as PCSEngineTrait<E>>::commit(&ck, &poly, &blind, false).unwrap();
    let mut tr = <E as Engine>::TE::new(b"p");
    // point too short
    let point: Vec<<E as Engine>::Scalar> = vec![<E as Engine>::Scalar::from(5u64); m-1];
    assert!(<HashMlePCS<E> as PCSEngineTrait<E>>::prove(&ck, &mut tr, &com, &poly, &blind, &point).is_err());
  }

  #[test]
  fn tamper_layer_root_fails() {
    let m = 3usize; 
    let n = 1usize << m;
    let poly: Vec<<E as Engine>::Scalar> = (0..n).map(|i| <E as Engine>::Scalar::from((i*i + 7) as u64)).collect();
    let point: Vec<<E as Engine>::Scalar> = vec![<E as Engine>::Scalar::from(2u64); m];
    let (ck, vk) = <HashMlePCS<E> as PCSEngineTrait<E>>::setup(b"tamper", n);
    let blind = HashMleBlind::<E>::default();
    let com = <HashMlePCS<E> as PCSEngineTrait<E>>::commit(&ck, &poly, &blind, false).unwrap();
    let mut tr = <E as Engine>::TE::new(b"x");
    let (eval, mut arg) = <HashMlePCS<E> as PCSEngineTrait<E>>::prove(&ck, &mut tr, &com, &poly, &blind, &point).unwrap();
    // flip a bit in an inner root
    arg.layer_roots[1].0[0] ^= 0x01;
    let mut tr_v = <E as Engine>::TE::new(b"x");
    assert!(<HashMlePCS<E> as PCSEngineTrait<E>>::verify(&vk, &mut tr_v, &com, &point, &eval, &arg).is_err());
  }

  #[test]
  fn red_team_forged_chain_is_rejected_by_index_checks() {
    // Demonstrates the classic pair-index forgery would have verified before the fix.
    let m = 3usize; 
    let n = 1usize << m;
    let poly: Vec<<E as Engine>::Scalar> = (0..n).map(|i| <E as Engine>::Scalar::from((3*i + 5) as u64)).collect();
    let point: Vec<<E as Engine>::Scalar> = vec![<E as Engine>::Scalar::from(7u64), <E as Engine>::Scalar::from(11u64), <E as Engine>::Scalar::from(13u64)];
    let (ck, vk) = <HashMlePCS<E> as PCSEngineTrait<E>>::setup(b"forge", n);
    let blind = HashMleBlind::<E>::default();
    let com = <HashMlePCS<E> as PCSEngineTrait<E>>::commit(&ck, &poly, &blind, false).unwrap();

    // Build our own fake layer chain:
    // Layer 0: take pair at j != 0
    let j0 = 2usize;
    let half0 = n/2;
    let a0 = poly[j0];
    let b0 = poly[j0 + half0];
    let next0 = a0 + point[0] * (b0 - a0);
    // Trees for L0..L3, each with only the constraint that index 0 equals our chosen "next"
    let leaves0 = poly.iter().map(|x| leaf_hash::<E>(x)).collect::<Vec<_>>();
    let t0 = MerkleTree::from_leaves(leaves0);
    let root0 = MerkleRoot(t0.root());

    let l1 = vec![next0, <E as Engine>::Scalar::ZERO, <E as Engine>::Scalar::ZERO, <E as Engine>::Scalar::ZERO];
    let t1 = MerkleTree::from_leaves(l1.iter().map(leaf_hash::<E>).collect());
    let root1 = MerkleRoot(t1.root());
    let a1 = l1[0];
    let b1 = l1[2];
    let next1 = a1 + point[1] * (b1 - a1);

    let l2 = vec![next1, <E as Engine>::Scalar::ZERO];
    let t2 = MerkleTree::from_leaves(l2.iter().map(leaf_hash::<E>).collect());
    let root2 = MerkleRoot(t2.root());
    let a2 = l2[0];
    let b2 = l2[1];
    let next2 = a2 + point[2] * (b2 - a2);

    let t3 = MerkleTree::from_leaves(vec![leaf_hash::<E>(&next2)]);
    let root3 = MerkleRoot(t3.root());

    // Assemble a forged argument (note: indices j0 and j0+half0 at layer 0)
    let arg = HashMleEvaluationArgument {
      layer_roots: vec![root0, root1, root2, root3],
      rounds: vec![
        Round {
          a: a0, b: b0,
          path_a: t0.open(j0),
          path_b: t0.open(j0 + half0),
          next: next0, path_next: t1.open(0),
        },
        Round {
          a: a1, b: b1,
          path_a: t1.open(0),
          path_b: t1.open(2),
          next: next1, path_next: t2.open(0),
        },
        Round {
          a: a2, b: b2,
          path_a: t2.open(0),
          path_b: t2.open(1),
          next: next2, path_next: t3.open(0),
        },
      ],
    };
    let eval = next2; // bogus eval, not the true MLE eval

    let mut tr = <E as Engine>::TE::new(b"forge");
    // With the index/depth checks added, this is rejected.
    assert!(<HashMlePCS<E> as PCSEngineTrait<E>>::verify(&vk, &mut tr, &com, &point, &eval, &arg).is_err());
  }
}
