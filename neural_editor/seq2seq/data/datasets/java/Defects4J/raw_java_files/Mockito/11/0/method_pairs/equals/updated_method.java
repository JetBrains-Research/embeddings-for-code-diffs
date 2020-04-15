
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o instanceof DelegatingMethod) {
            DelegatingMethod that = (DelegatingMethod) o;
            return method.equals(that.method);
        } else {
            return method.equals(o);
        }
    }