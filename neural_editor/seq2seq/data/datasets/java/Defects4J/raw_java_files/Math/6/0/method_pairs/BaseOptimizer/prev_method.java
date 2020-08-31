
    protected BaseOptimizer(ConvergenceChecker<PAIR> checker) {
        this.checker = checker;

        evaluations = new Incrementor(0, new MaxEvalCallback());
        iterations = new Incrementor(0, new MaxIterCallback());
    }