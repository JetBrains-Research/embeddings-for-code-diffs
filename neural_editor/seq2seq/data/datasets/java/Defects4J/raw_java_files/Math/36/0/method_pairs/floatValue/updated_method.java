
    @Override
    public float floatValue() {
        float result = numerator.floatValue() / denominator.floatValue();
        if (Double.isNaN(result)) {
            // Numerator and/or denominator must be out of range:
            // Calculate how far to shift them to put them in range.
            int shift = Math.max(numerator.bitLength(),
                                 denominator.bitLength()) - Float.MAX_EXPONENT;
            result = numerator.shiftRight(shift).floatValue() /
                denominator.shiftRight(shift).floatValue();
        }
        return result;
    }