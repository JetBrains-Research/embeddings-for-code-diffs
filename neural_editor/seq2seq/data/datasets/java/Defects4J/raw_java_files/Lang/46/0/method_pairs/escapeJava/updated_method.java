
    public static void escapeJava(Writer out, String str) throws IOException {
        escapeJavaStyleString(out, str, false, false);
    }