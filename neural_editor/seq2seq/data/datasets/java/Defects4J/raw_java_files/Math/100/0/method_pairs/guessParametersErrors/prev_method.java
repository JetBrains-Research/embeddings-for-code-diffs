
    public double[] guessParametersErrors(EstimationProblem problem)
      throws EstimationException {
        int m = problem.getMeasurements().length;
        int p = problem.getAllParameters().length;
        if (m <= p) {
            throw new EstimationException("no degrees of freedom ({0} measurements, {1} parameters)",
                                          new Object[] { new Integer(m), new Integer(p)});
        }
        double[] errors = new double[problem.getAllParameters().length];
        final double c = Math.sqrt(getChiSquare(problem) / (m - p));
        double[][] covar = getCovariances(problem);
        for (int i = 0; i < errors.length; ++i) {
            errors[i] = Math.sqrt(covar[i][i]) * c;
        }
        return errors;
    }