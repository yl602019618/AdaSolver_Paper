# Prepared-request timing scope

the per-request wall-time observations begin with normalized inputs and reusable geometry already resident on the host. File loading, normalization fitting and geometry construction used to prepare a case are outside that window; point selection, correction when applicable, device transfers, model forward and requested full-field/force output are inside. These prepared-request observations must not be presented as cold-input latency or substituted for E1.

This clarifies the implemented boundary without changing execution or numerical results. The frozen protocol file is preserved.
