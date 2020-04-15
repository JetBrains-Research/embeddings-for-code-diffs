
        @Override
        protected Class<?> resolveClass(ObjectStreamClass desc) throws IOException, ClassNotFoundException {
            String name = desc.getName();
            try {
                return Class.forName(name, false, classLoader);
            } catch (ClassNotFoundException ex) {
                    return Class.forName(name, false, Thread.currentThread().getContextClassLoader());
            }
        }