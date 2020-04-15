
    public Complex(double real, double imaginary) {
        this.real = real;
        this.imaginary = imaginary;

        isNaN = Double.isNaN(real) || Double.isNaN(imaginary);
        isInfinite = !isNaN &&
            (Double.isInfinite(real) || Double.isInfinite(imaginary));
        isZero = real == 0 && imaginary == 0;
    }