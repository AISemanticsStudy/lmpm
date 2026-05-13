# LMPM

## TL;DR
LMPM provides PyTorch with distributed programming abstractions, much as Kubernetes provides container orchestration, but at the computation-graph level.

## What is LMPM?
LMPM (latent message passing machine) is a programming language and runtime for Consensual Programming. At the time of its creation,
LMPM was the first system of its kind, developed in response to the growing complexity of modern deep learning systems.

The primary goal of LMPM is to enable the construction of AI-like systems with **fine-grained control over inductive bias** while greatly
reducing the complexity and cognitive burden of manually implementing computation and communication.
LMPM is revolutionary in the sense that it makes inductive bias a first-class concept in programming.
It is designed for the current generation of deep learning while also providing tools for future generations. With that said, we don't intend to make LMPM a general-purpose programming language.
LMPM is designed to be a domain-specific language tailored for building AI-like systems.

LMPM implements a variant of the actor model where state transitions are derived from constraint satisfaction rather than manually programmed. Actors (latent objects) exchange messages, but their behavior emerges from the consensus rules rather than explicit state machines.

## Latent message passing vibe
You might never have heard of "latent message passing" before.
But once you have seen its elegance, you can't unsee it.

Here is how we represent GPT's causal mask in LMPM: position `t` can only see strictly earlier tokens.
```
visible Txt[t] <- Txt[0..t-1] when t > 0
```
And that's it.

You no longer need to carry the cognitive load of constructing and reasoning about different
kinds of attention masks as inductive bias, or their implications, in your head.

## Comparisons to other AI systems

The only fundamental difference between LMPM and previous ML frameworks and runtimes
is that:

LMPM is **not** a computation graph framework.

LMPM's compiler produces an actor state-machine transition loop for each atomic region of state evolution in the whole system.
In other words:
- only the LMPM resolver or compiler has a ground-truth view of the world
- each PyTorch worker managed by the LMPM runtime has only a local view of the world and does not need to know the roles or responsibilities of the other workers in the same collective communication group.

The LMPM runtime is responsible only for routing messages correctly between workers, with correctness guaranteed by the underlying logical clock.

While LMPM prioritizes correctness and abstraction, its architecture enables performance through:
- Decoupled coordination and computation planes
- Direct worker-to-worker communication bypassing the runtime
- Optimized routing based on logical clock dependencies

LMPM is designed to be **correct by construction and abstraction** while maintaining practical performance for distributed AI systems.

So even a seasoned AI researcher might find LMPM off-putting at first glance.
Once the syntax barrier is crossed, you will find that LMPM has the right separation of concerns between:
- system-state evolution logic
- inductive bias definition and enforcement
- distributed execution

and that it directly addresses the pain points in building gigantic AI-like systems and developing new ML systems with
complex yet reasonable inductive biases.

## Consensual Programming, introduced by LMPM
Consensual Programming is a programming paradigm in which
system state evolves only through proposals evaluated under explicit constraints and committed by agreement,
rather than by imperative execution.

To put that in perspective:
- Functional Programming says: don’t mutate values.
- Consensual Programming says: don’t mutate reality. Reality only changes by agreement.

In LMPM, everything goes through:
```sql
compute → propose → check → agree → commit
```

## What is an LMPM program?
An LMPM program is an (ultra) high-level description of a distributed system in terms of latent message passing.
If we have to coin a term, we can say that an LMPM program is a DBD-SCS (database-driven sampling control system).
Usually, an LMPM program consists of:
- value definitions
```
value Txt : Token[BPE] shape Seq(N) clock Lamport
value Vid : Token[VQ]  shape Grid3D(T, W, H) clock Lamport
value MaskPlan : Event shape Set() clock Lamport  // Scheduling log (CRDT)
value RuleSet  : Rule shape Set() clock Lamport   // Constraint set (CRDT)
```
- visibility definitions
```
visible Vid[t,x,y] <- Vid[t,x+dx,y+dy] when (dx,dy) in N8
visible Vid[t,x,y] <- Vid[t-1,x,y] when t>0
visible Vid[t,x,y] <- Txt[*]        // All captions visible
value Coarse : Token[VQ] shape Grid3D(T, W/8, H/8) clock Lamport
visible Vid[t,x,y] <- Coarse[t, x/8, y/8]
```
- rule definitions to generate proposals
```
rule LocalGen:
  on Vid[t,x,y]
  read {neighbors in N8, prev_frame, Txt[*], Coarse?}
  emit proposal Vid[t,x,y] := Model(neighbors, prev, txt, coarse)
  score ModelScore(...)
```
- constraint definitions to introduce inductive bias
```
constraint HardLock:
  forbid change(Vid[t, x1..x2, y1..y2]) priority 100

constraint TextPaint:
  require Vid[t, region("sky")] in Palette("blue-ish") priority 50

constraint Topology:
  forbid read_future(Vid[t,x,y]) priority 1000
```
- commit rule definitions to reach agreement
```
commit Vid[*] using satisfy(max_constraints_then_score)
```
- scheduling rule definitions
```
rule Schedule:
  on MaskPlan
  emit proposal MaskPlan := SampleTiles(strategy="red-black", epoch=e)
```

## Design principles of LMPM
LMPM is conservative about what becomes true.
It is liberal about what can be attempted.

## LMPM’s syntax is inspired by databases, logic, and actor systems
LMPM’s syntax comes from treating causality, agreement, and visibility as first-class syntax, borrowing surface forms from databases, logic, and actor systems—but rejecting their execution-centric assumptions.

## Because it is causal, LMPM's syntax is also designed for AI chatbots
LMPM's syntax has a strict lexical causal structure designed to be both human-readable and machine-parsable. This enables AI assistants to reason about and generate LMPM code more effectively than they can with traditional imperative code.

## LMPM is debuggable at the constraint level
The LMPM runtime tracks the constraint evaluation results for each commit attempt and reports them back to the user.
This enables the user to debug the model at the constraint level instead of at the tensor-operation level.
This is a game-changer when building AI-like systems with complex inductive bias.

## LMPM abstracts gradient computation through constraint satisfaction rather than explicit `.backward()` calls
In LMPM, backpropagation (BP) is internalized as a constraint satisfaction problem. When the resolver recognizes a pattern (DAG + differentiable ops + desired exactness), it can lower it to optimized BP kernels.

## Inductive bias in LMPM
In a behavior-centric view, inductive bias is the operator itself, not an operand in an arithmetic-centric view.
LMPM makes inductive bias explicit, programmable, and debuggable.
Inductive bias is treated as the real operator of an AI system built with LMPM.
Compared with other AI systems that hide inductive bias as operands or parameters, one unique aspect of LMPM is that we preserve the same level of
fine-grained control and observability over inductive bias at every level of the stack, from language semantics, IR, resolver, and compiler to runtime.

## Commit in LMPM
In LMPM, commit is the sole causal interface between the system and the external world.
All intermediate states may be observed internally, but have no external effect until committed.

## Anatomy

We structure the system by separating the **Logic of Agreement** (Control Plane) from the **Physics of Computation** (Data Plane).

| Layer | Component | Responsibility | Who controls it |
| --- | --- | --- | --- |
| **Semantics** | LMPM Frontend | Syntax & Semantics | You / The Spec |
| **Logic** | Lamport IR | Causal Dependencies | Standard / Ecosystem |
| **Policy** | **Resolver** | Routing & Consensus Strategy | **Research Lab / Product** |
| **Orchestration** | LMPM Runtime | Control Plane (Signaling) | Elixir (BEAM) |
| **Execution** | Workers | Data Plane (Computation) | PyTorch / NCCL / Triton |

### LMPM Frontend

The LMPM Frontend is a programming language front end for expressing consensual programming semantics. It compiles source code into **Lamport IR**, preserving the lexical causal structure defined by the user.

### Lamport IR (Logical Clock Intermediate Representation)

A language-agnostic IR designed to express consensual programming semantics. It creates a reified graph of:

* **Latent Messages** (Data-flow intent)
* **Proposals** (Tentative state)
* **Constraints** (Inductive bias)
* **Commits** (Finalized reality)

Crucially, Lamport IR describes *what* must happen and *when* (logically), but not *where* or *how*.

### LMPM Resolver

The Resolver is the policy engine. It lowers the abstract Lamport IR into concrete execution plans. It decides:

* **Topology:** Which worker computes which tile of `Vid[t,x,y]`.
* **Routing:** Who needs to send data to whom to satisfy `visible` clauses.
* **Resolution:** How to handle conflicting proposals (e.g., `average`, `winner-takes-all`).

If a behavior difference cannot be explained as a resolver policy choice over the same IR, it is a language **bug**.

### LMPM Runtime (The Control Plane)

We use **Elixir/OTP** (running on BEAM) to implement the Runtime because the actor model is well suited to managing the lifecycle of distributed agreement.

* **Role:** The Runtime is the "Air Traffic Controller." It manages the **Signaling** (Proposals, Commits, Permissions).
* **Constraint:** The Runtime **never touches tensors**. It only handles metadata (shapes, locations, checksums, logical timestamps).

### LMPM Workers (The Data Plane)

The actual computation happens in PyTorch processes.

* **Role:** The Workers are the "Planes." They carry the heavy payload.
* **Execution:** When the Runtime approves a `Commit`, it instructs Workers to execute the compute kernels.
* **Transport:** When the Runtime determines that a message needs to pass from Worker A to Worker B, it establishes a high-performance side channel (e.g., NCCL, shared memory, NVLink). The data bypasses the Elixir Runtime entirely.

### Redundancy and Elasticity

Redundancy and elasticity are possible because the Runtime (Elixir) holds the ground truth of the logical clock, while Workers are stateless functional units relative to the consensus:

* **Elasticity:** A Worker can crash and restart. The Runtime simply reroutes the "Compute Proposal" request to a new worker.
* **Redundancy:** The Resolver can instruct three Workers to propose the same `Vid[t]` value. The Runtime waits for a quorum before committing. This is managed purely via Elixir message passing, with zero overhead on the GPU compute path.

## Author

Shenghang Cai ([ORCID: 0009-0003-7397-1203](https://orcid.org/0009-0003-7397-1203)).

## Citation

If you use LMPM, please cite it using the metadata in [CITATION.cff](CITATION.cff):

```bibtex
@software{cai_lmpm,
  author = {Cai, Shenghang},
  title = {LMPM},
  url = {https://github.com/AISemanticsStudy/lmpm}
}
```
