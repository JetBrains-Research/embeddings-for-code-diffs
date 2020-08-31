
    public double getRMS() {
        double criterion = 0;
        for (int i = 0; i < rows; ++i) {
            final double residual = residuals[i];
            criterion += residual * residual * residualsWeights[i];
        }
        return Math.sqrt(criterion / rows);
    }