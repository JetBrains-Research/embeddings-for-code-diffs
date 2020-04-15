
    public static int lcm(int a, int b) {
        if (a==0 || b==0){
            return 0;
        }
        int lcm = Math.abs(mulAndCheck(a / gcd(a, b), b));
        if (lcm == Integer.MIN_VALUE){
            throw new ArithmeticException("overflow: lcm is 2^31");
        }
        return lcm;
    }