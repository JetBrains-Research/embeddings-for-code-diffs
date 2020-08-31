
  public static Vector3D crossProduct(final Vector3D v1, final Vector3D v2) {

      final double n1 = v1.getNormSq();
      final double n2 = v2.getNormSq();
      if ((n1 * n2) < MathUtils.SAFE_MIN) {
          return ZERO;
      }

      // rescale both vectors without losing precision,
      // to ensure their norm are the same order of magnitude
      final int deltaExp = (FastMath.getExponent(n1) - FastMath.getExponent(n2)) / 4;
      final double x1    = FastMath.scalb(v1.x, -deltaExp);
      final double y1    = FastMath.scalb(v1.y, -deltaExp);
      final double z1    = FastMath.scalb(v1.z, -deltaExp);
      final double x2    = FastMath.scalb(v2.x,  deltaExp);
      final double y2    = FastMath.scalb(v2.y,  deltaExp);
      final double z2    = FastMath.scalb(v2.z,  deltaExp);

      // we reduce cancellation errors by preconditioning,
      // we replace v1 by v3 = v1 - rho v2 with rho chosen in order to compute
      // v3 without loss of precision. See Kahan lecture
      // "Computing Cross-Products and Rotations in 2- and 3-Dimensional Euclidean Spaces"
      // available at http://www.cs.berkeley.edu/~wkahan/MathH110/Cross.pdf

      // compute rho as an 8 bits approximation of v1.v2 / v2.v2
      final double ratio = (x1 * x2 + y1 * y2 + z1 * z2) / FastMath.scalb(n2, 2 * deltaExp);
      final double rho   = FastMath.rint(256 * ratio) / 256;

      final double x3 = x1 - rho * x2;
      final double y3 = y1 - rho * y2;
      final double z3 = z1 - rho * z2;

      // compute cross product from v3 and v2 instead of v1 and v2
      return new Vector3D(y3 * z2 - z3 * y2, z3 * x2 - x3 * z2, x3 * y2 - y3 * x2);

  }