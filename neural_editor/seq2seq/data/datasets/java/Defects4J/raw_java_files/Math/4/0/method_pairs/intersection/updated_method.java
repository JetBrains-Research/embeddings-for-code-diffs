
    public Vector3D intersection(final SubLine subLine, final boolean includeEndPoints) {

        // compute the intersection on infinite line
        Vector3D v1D = line.intersection(subLine.line);
        if (v1D == null) {
            return null;
        }

        // check location of point with respect to first sub-line
        Location loc1 = remainingRegion.checkPoint(line.toSubSpace(v1D));

        // check location of point with respect to second sub-line
        Location loc2 = subLine.remainingRegion.checkPoint(subLine.line.toSubSpace(v1D));

        if (includeEndPoints) {
            return ((loc1 != Location.OUTSIDE) && (loc2 != Location.OUTSIDE)) ? v1D : null;
        } else {
            return ((loc1 == Location.INSIDE) && (loc2 == Location.INSIDE)) ? v1D : null;
        }

    }