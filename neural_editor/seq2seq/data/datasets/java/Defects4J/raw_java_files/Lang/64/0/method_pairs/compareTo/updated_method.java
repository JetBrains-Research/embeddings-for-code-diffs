
    public int compareTo(Object other) {
        if (other == this) {
            return 0;
        }
        if (other.getClass() != this.getClass()) {
            if (other.getClass().getName().equals(this.getClass().getName())) {
                return iValue - getValueInOtherClassLoader(other);
            }
            throw new ClassCastException(
                    "Different enum class '" + ClassUtils.getShortClassName(other.getClass()) + "'");
        }
        return iValue - ((ValuedEnum) other).iValue;
    }