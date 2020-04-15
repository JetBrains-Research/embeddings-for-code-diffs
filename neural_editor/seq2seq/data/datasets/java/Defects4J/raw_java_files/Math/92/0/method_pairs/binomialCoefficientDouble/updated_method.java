
    public static double binomialCoefficientDouble(final int n, final int k) {
        if (n < k) {
            throw new IllegalArgumentException(
                "must have n >= k for binomial coefficient (n,k)");
        }
        if (n < 0) {
            throw new IllegalArgumentException(
                "must have n >= 0 for binomial coefficient (n,k)");
        }
        if ((n == k) || (k == 0)) {
            return 1d;
        }
        if ((k == 1) || (k == n - 1)) {
            return n;
        }
        if (k > n/2) {
            return binomialCoefficientDouble(n, n - k);
        }
        if (n < 67) {
            return binomialCoefficient(n,k);
        }
        
        double result = 1d;
        for (int i = 1; i <= k; i++) {
             result *= (double)(n - k + i) / (double)i;
        }
  
        return Math.floor(result + 0.5);
    }