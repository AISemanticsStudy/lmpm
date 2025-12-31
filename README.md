# LMPM

LMPM(latent message passing machine) is a programming language and runtime for Consensual Programming. At the moment when it is created,
it is first of of its kind. It was developed after modern deep learning reaching its complexity limits in around 2025.

The primary goal of LMPM is to enable building AI-like systems with fine-grained control in inductive bias while greatly
clamping the complicity in manually implement computation and communication. LMPM is revolutionary in the sense that it pushes
the inductive bias to a first-class citizen in programming, and it is designed not just for current generation of DL but also
provide the tools for future generations. LMPM might sounds over-promising, and frivolously so, because it operates at a
much higher level of abstraction than any other programming languages, and operates at such a high volume of data and computation
capacity. With that being said, we don't intend to make LMPM a general-purpose programming language.
Instead, LMPM is designed to be a domain-specific language for building AI-like systems.

LMPM is a faithful implementation of the actor model but it operates at latent message passing between latten objects as senders and receivers
instead of human-hard-coded state machine.

## Consensual Programming
Consensual Programming is a programming paradigm in which
system state evolves only through proposals evaluated under explicit constraints and committed by agreement,
rather than by imperative execution.

To put that in perspective:
- Functional Programming says: don’t mutate values.
- Consensual Programming says: don’t mutate reality. Reality only changes by agreement.

## What is a LMPM program?
A LMPM program is a (ultra) high-level description of a distributed system in terms of latent message passing.
Usually a LMPM program consists of:
- value definitions
```
value Txt : Token[BPE] shape Seq(N) clock Lamport
value Vid : Token[VQ]  shape Grid3D(T, W, H) clock Lamport
value MaskPlan : Event shape Set() clock Lamport  // 调度日志（CRDT）
value RuleSet  : Rule shape Set() clock Lamport   // 约束集合（CRDT）
```
- visibility definitions
```
visible Vid[t,x,y] <- Vid[t,x+dx,y+dy] when (dx,dy) in N8
visible Vid[t,x,y] <- Vid[t-1,x,y] when t>0
visible Vid[t,x,y] <- Txt[*]        // 全 caption 可见
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
- commit rules definitions to reach agreement
```
commit Vid[*] using satisfy(max_constraints_then_score)
```
- rule definitions to schedule the system
```
rule Schedule:
  on MaskPlan
  emit proposal MaskPlan := SampleTiles(strategy="red-black", epoch=e)
```

## Anatomy
- LMPM frontend: a programming language frontend to express consensual programming semantics. It can be embedded in other programming languages like Python and send the IR to LMPM gateway for compilation and execution. LMPM compiler shipped with a native frontend in Elixir leveraging LISP style meta-programming.
- LCI IR (logical clock indexing intermediate representation): an language agnostic (IR) specific design to express consensual programming semantics with regard to logical clock, including latent message passing, proposals, constraints, and commits(agreements).
- LMPM compiler: compile the IR to PyTorch source code for execution. We use PyTorch to implement LMPM compiler's first backend target because of its rich ecosystem in deep learning.
It is open to implement other backends in the future.
- LMPM runtime: an PyTorch process manager to run compiled LMPM programs with distributed execution support. We chose to use Elixir to implement LMPM runtime because of it runs on BEAM which is a robust actor model runtime, which save us a lot of effort from implementing a new actor model runtime.
- LMPM gateway: an API server to receive inference requests or IR submissions from other programming languages.

## Collective communication in LMPM
The compiler and runtime of LMPM natively support collective communication patterns including broadcast, scatter, gather, reduce, all-reduce, all-gather, and scatter-gather. Note that in LMPM we try our best to decouple collective communication (worker↔worker / GPU↔GPU) from the control plane (runtime↔worker).

## Comparisons to other AI systems

### Compared to PyTorch

### Compared to DL runtimes of a static graph
