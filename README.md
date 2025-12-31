# LMPM

LMPM(latent message passing machine) is a programming language and runtime for Consensual Programming. At the moment when it is created,
it is first of of its kind. It was developed after modern deep learning reaching its complexity limits in around 2025.

The primary goal of LMPM is to enable building AI-like systems with fine-grained control in inductive bias while greatly
clamping the complicity in computation and communication.

LMPM is a faithful implementation of the actor model but it operates at latent message passing between latten objects as senders and receivers
instead of human-hard-coded state machine.

## Consensual Programming
Consensual Programming is a programming paradigm in which
system state evolves only through proposals evaluated under explicit constraints and committed by agreement,
rather than by imperative execution.

To put that in perspective:
- Functional Programming says: don’t mutate values.
- Consensual Programming says: don’t mutate reality. Reality only changes by agreement.

## Anatomy
- LMPM frontend: a programming language frontend to express consensual programming semantics. It can be embedded in other programming languages like Python and send the IR to LMPM gateway for compilation and execution. LMPM compiler shipped with a native frontend in Elixir leveraging LISP style meta-programming.
- LCI IR (logical clock indexing intermediate representation): an language agnostic (IR) specific design to express consensual programming semantics with regard to logical clock, including latent message passing, proposals, constraints, and commits(agreements).
- LMPM compiler: compile the IR to PyTorch source code for execution. We use PyTorch to implement LMPM compiler's first backend target because of its rich ecosystem in deep learning.
It is open to implement other backends in the future.
- LMPM runtime: an PyTorch process manager to run compiled LMPM programs with distributed execution support. We chose to use Elixir to implement LMPM runtime because of it runs on BEAM which is a robust actor model runtime, which save us a lot of effort from implementing a new actor model runtime.
- LMPM gateway: an API server to receive inference requests or IR submissions from other programming languages.
