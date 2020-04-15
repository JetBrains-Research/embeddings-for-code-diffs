
    public LevenbergMarquardtOptimizer() {

        // set up the superclass with a default  max cost evaluations setting
        setMaxIterations(1000);

        // default values for the tuning parameters
        setInitialStepBoundFactor(100.0);
        setCostRelativeTolerance(1.0e-10);
        setParRelativeTolerance(1.0e-10);
        setOrthoTolerance(1.0e-10);

    }