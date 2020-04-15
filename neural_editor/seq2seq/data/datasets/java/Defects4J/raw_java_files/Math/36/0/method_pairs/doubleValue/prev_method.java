
    @Override
    public double doubleValue() {
        double result = numerator.doubleValue() / denominator.doubleValue();
            // Numerator and/or denominator must be out of range:
            // Calculate how far to shift them to put them in range.
        return result;
    }