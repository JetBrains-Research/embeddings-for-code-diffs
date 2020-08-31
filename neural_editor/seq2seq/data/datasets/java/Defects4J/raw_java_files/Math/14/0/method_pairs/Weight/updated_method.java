
    public Weight(double[] weight) {
        final int dim = weight.length;
        weightMatrix = new DiagonalMatrix(weight);
    }