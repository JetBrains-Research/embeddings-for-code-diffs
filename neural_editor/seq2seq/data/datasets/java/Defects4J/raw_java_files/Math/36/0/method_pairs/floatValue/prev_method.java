
    @Override
    public float floatValue() {
        float result = numerator.floatValue() / denominator.floatValue();
            // Numerator and/or denominator must be out of range:
            // Calculate how far to shift them to put them in range.
        return result;
    }