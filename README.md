# LMPM

LMPM(latent message passing machine) is a programming language and runtime for Consensual Programming. At the moment when it is created,
it is first of of its kind. It was developed after modern deep learning reaching its complexity limits in around 2025.

The primary goal of LMPM is to enable building AI-like systems with fine-grained control in inductive bias while greatly
clamping the complicity and intellectual challenges in manually implementing computation and communication.
LMPM is revolutionary in the sense that it pushes the inductive bias to a first-class citizen in programming,
and it is designed not just for current generation of DL but also
provide the tools for future generations. LMPM might sounds over-promising, and frivolously so, because it operates at a
much higher level of abstraction than any other programming languages, and operates at such a high volume of data and computation
capacity. With that being said, we don't intend to make LMPM a general-purpose programming language.
Instead, LMPM is designed to be a domain-specific language for building AI-like systems.

LMPM is a faithful implementation of the actor model but it operates at latent message passing between latten objects as senders and receivers
instead of human-hard-coded state machine.

## Comparisons to other AI systems

The only fundamental difference between LMPM and precedent ML frameworks, runtimes
is that:

LMPM is **not** a computation graph framework.

LMPM's compiler will produce a actor state machine-transition loop for each atomic evolvement area of the whole system.
In other words
- only LMPM's resolver or compiler has a ground truth world view
- each Pytorch worker managed by LMPM runtime only has a local view of the world,
and has no interests in knowing the roles and responsibilities of other workers it is conducting collective communication with.

LMPM runtime is only responsible for routing messages between workers correctly, guaranteed by the underlying logical clock.

It is worth mentioning that LMPM is **not**:
- designed this way to be performant
- designed this way to be scalable
- designed this way to be easy to use

Instead, LMPM is designed this way to be **correct by construction and abstraction**.
So even a seasoned AI research might find LMPM to be off-putting at first glance.
Once the syntax barrier is crossed, you would find LMPM has exactly right separation of concerns between:
- system state evolvement logic
- inductive bias definition and enforcement
- distributed execution

And perfectly heal the pain points in building gigantic AI-like systems and developing an new ML with
complex yet reasonable inductive bias.

## Consensual Programming
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

## What is a LMPM program?
A LMPM program is an (ultra) high-level description of a distributed system in terms of latent message passing.
But if we have to coin one terminology, we can say that a LMPM program is a DBD-SCS (database-driven sampling control system).
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

## Being causal, LMPM's syntax is designed for AI chatbot as well
LMPM source code is designed to be fed to an AI chatbot.
The take is that having an AI model defined in a super high-level consensual programming language,
with explicit definitions of inductive bias and system evolvement rules,
would greatly reduce the complexity of the AI model implementation,
by exploiting AI chatbot exceptional reasoning ability on a very structured and high-level codebase.
In other words, LMPM's syntax has a **strict lexical causal structure**.
You can think of it as a frantic cousin of Rust's ownership system,
but applied to logical clock and sound textual reasoning.

## LMPM debuggable at constraint level
LMPM runtime tracks the constraint evaluation results at each commit attempt, and report them back to the user.
This enables the user to debug the model at constraint level instead of at tensor operation level.
This is a game changer in building AI-like systems with complex inductive bias.

## There is no `.backward()` in LMPM
LMPM views BP as computing a specific fixed point.
BP is one message passing algorithm that solves a constraint system (“chain rule consistency”) efficiently.
In LMPM can BP is internalized: when the resolver recognizes a pattern (DAG + differentiable ops + desired exactness), it can lower to BP kernels.

## LMPM's three categories of mechanics
LMPM's mechanics can be categorized into three orthogonal dimensions:

| Category      | Lives in LMPM                     | Examples               |
| ------------- | --------------------------------- | ---------------------- |
| **Semantics** | visibility / constraints / commit | masks, topology, locks |
| **Solver**    | runtime / compiler passes         | LR, BP, optimizer      |
| **Capacity**  | rule internals                    | heads, width, depth    |

## Anatomy

We put resolver as a first-class citizen in the toolchain:

| Layer                          | Who controls it                           |
| ------------------------------ | ----------------------------------------- |
| LMPM language semantics        | You / the spec                            |
| LCI IR                         | Ecosystem / standard                      |
| **Resolver**                   | **Organization / product / research lab** |
| Backend (PyTorch, XLA, Triton) | Infrastructure                            |


### LMPM frontend
a programming language frontend to express consensual programming semantics. It can be embedded in other programming languages like Python and send the IR to LMPM gateway for compilation and execution. LMPM compiler shipped with a native frontend in Elixir leveraging LISP style meta-programming.

### LCI IR (logical clock indexing intermediate representation)
an language agnostic (IR) specific design to express consensual programming semantics with regard to logical clock, including latent message passing, proposals, constraints, and commits(agreements).

### LMPM resolver
In LMPM, the resolver is a first-class, programmable component implemented in external languages. It operates as a policy engine over a fixed intermediate representation, deciding routing, scheduling, clock interpretation, and agreement strategies without altering program semantics. This separation allows different organizations to deploy resolvers optimized for their needs—performance, safety, determinism—while sharing the same language, inductive bias, and model identity.

If a behavior difference cannot be explained as a resolver policy choice over the same IR, it is a language **bug**.

```
Resolver API
------------
on_proposals(proposals, log_state) -> routing_plan
on_conflict(target, competing_proposals) -> resolution
on_commit(candidate_set) -> commit_decision
on_clock_event(event) -> ordering_decision
```

### LMPM compiler
compile the IR to PyTorch source code for execution. We use PyTorch to implement LMPM compiler's first backend target because of its rich ecosystem in deep learning.
It is open to implement other backends in the future.

### LMPM runtime
an PyTorch process manager to run compiled LMPM programs with distributed execution support. We chose to use Elixir to implement LMPM runtime because of it runs on BEAM which is a robust actor model runtime, which save us a lot of effort from implementing a new actor model runtime.

### LMPM gateway
an API server to receive inference requests or IR submissions from other programming languages.

## Collective communication in LMPM
The compiler and runtime of LMPM natively support collective communication patterns including broadcast, scatter, gather, reduce, all-reduce, all-gather, and scatter-gather. Note that in LMPM we try our best to decouple collective communication (worker↔worker / GPU↔GPU) from the control plane (runtime↔worker).
