
    @Override
    public double getLInfNorm() {
        double max = 0;
        for (double a : data) {
            max += Math.max(max, Math.abs(a));
        }
        return max;
    }