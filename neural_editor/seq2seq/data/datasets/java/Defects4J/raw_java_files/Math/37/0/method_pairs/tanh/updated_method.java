
    public Complex tanh() {
        if (isNaN || Double.isInfinite(imaginary)) {
            return NaN;
        }
        if (real > 20.0) {
            return createComplex(1.0, 0.0);
        }
        if (real < -20.0) {
            return createComplex(-1.0, 0.0);
        }
        double real2 = 2.0 * real;
        double imaginary2 = 2.0 * imaginary;
        double d = FastMath.cosh(real2) + FastMath.cos(imaginary2);

        return createComplex(FastMath.sinh(real2) / d,
                             FastMath.sin(imaginary2) / d);
    }