
    public BrentOptimizer() {
        setMaxEvaluations(Integer.MAX_VALUE);
        setMaximalIterationCount(100);
        setAbsoluteAccuracy(1E-10);
        setRelativeAccuracy(1.0e-14);
    }