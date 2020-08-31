
    private void computeGershgorinCircles() {

        final int m     = main.length;
        final int lowerStart = 4 * m;
        final int upperStart = 5 * m;
        lowerSpectra = Double.POSITIVE_INFINITY;
        upperSpectra = Double.NEGATIVE_INFINITY;
        double eMax = 0;

        double eCurrent = 0;
        for (int i = 0; i < m - 1; ++i) {

            final double dCurrent = main[i];
            final double ePrevious = eCurrent;
            eCurrent = Math.abs(secondary[i]);
            eMax = Math.max(eMax, eCurrent);
            final double radius = ePrevious + eCurrent;

            final double lower = dCurrent - radius;
            work[lowerStart + i] = lower;
            lowerSpectra = Math.min(lowerSpectra, lower);

            final double upper = dCurrent + radius;
            work[upperStart + i] = upper;
            upperSpectra = Math.max(upperSpectra, upper);

        }

        final double dCurrent = main[m - 1];
        final double lower = dCurrent - eCurrent;
        work[lowerStart + m - 1] = lower;
        lowerSpectra = Math.min(lowerSpectra, lower);
        final double upper = dCurrent + eCurrent;
        work[upperStart + m - 1] = upper;
        upperSpectra = Math.max(upperSpectra, upper);
        minPivot = MathUtils.SAFE_MIN * Math.max(1.0, eMax * eMax);

    }