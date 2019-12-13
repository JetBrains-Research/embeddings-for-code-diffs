import com.github.gumtreediff.actions.ActionGenerator;
import com.github.gumtreediff.actions.model.Action;
import com.github.gumtreediff.gen.Generators;
import com.github.gumtreediff.matchers.Matcher;
import com.github.gumtreediff.matchers.Matchers;
import com.github.gumtreediff.tree.ITree;
import com.github.gumtreediff.tree.TreeContext;
import com.google.common.collect.Sets;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import static org.eclipse.jdt.core.dom.ASTNode.METHOD_DECLARATION;
import static org.eclipse.jdt.core.dom.ASTNode.SIMPLE_NAME;
import static org.eclipse.jdt.core.dom.ASTNode.JAVADOC;


class DiffExtractor {
    private Path root;
    private String prevFileText;
    private String updatedFileText;
    private TreeContext prevContext;
    private TreeContext updatedContext;
    private Matcher matcher;
    private Set<ITree> changedNodes;

    DiffExtractor(@NotNull Path root) throws IOException {
        this.root = root;
        Path prev = root.resolve("prev.java");
        Path updated = root.resolve("updated.java");
        prevFileText = Files.readString(prev);
        updatedFileText = Files.readString(updated);
        prevContext = getContextSafely(prev);
        updatedContext = getContextSafely(updated);
        matcher = Matchers.getInstance().getMatcher(prevContext.getRoot(), updatedContext.getRoot());
        matcher.match();

        ActionGenerator generator =
                new ActionGenerator(prevContext.getRoot(), updatedContext.getRoot(), matcher.getMappings());
        generator.generate();
        List<Action> actions = generator.getActions();
        changedNodes = actions.stream().map(Action::getNode).collect(Collectors.toUnmodifiableSet());
    }

    private TreeContext getContextSafely(Path filePath) throws IOException {
        try {
            return Generators.getInstance().getTree(filePath.toAbsolutePath().toString());
        } catch(RuntimeException e) {
            throw new IOException("Exception during parsing java file " + filePath.toAbsolutePath().toString() + ".", e);
        }
    }

    @NotNull
    private List<MethodDiff> extractMethodDiffs() {
        List<MethodDiff> methodDiffs = new ArrayList<>();
        extractMethodDiffsRecursively(prevContext.getRoot(), methodDiffs);
        return methodDiffs.stream().filter(MethodDiff::hasMethodChanged).collect(Collectors.toList());
    }

    private void extractMethodDiffsRecursively(ITree prev, List<MethodDiff> result) {
        ITree updated = matcher.getMappings().getDst(prev);
        if (prev.getType() == METHOD_DECLARATION && updated != null && updated.getType() == METHOD_DECLARATION) {
            result.add(new MethodDiff(prev, updated));
        }
        for (ITree child: prev.getChildren()) {
            extractMethodDiffsRecursively(child, result);
        }
    }

    boolean saveMethodDiffs() throws IOException {
        List<MethodDiff> methodDiffs = extractMethodDiffs();
        if (methodDiffs.isEmpty()) {
            return false;
        }
        Path currentDir = root.resolve("method_pairs");
        Files.createDirectories(currentDir);
        for (MethodDiff methodDiff: methodDiffs) {
            Path dirToWriteFiles = currentDir.resolve(methodDiff.getMethodName());
            Files.createDirectories(dirToWriteFiles);
            Files.writeString(dirToWriteFiles.resolve("prev_method.java"), methodDiff.getPrev());
            Files.writeString(dirToWriteFiles.resolve("updated_method.java"), methodDiff.getUpdated());
        }
        return true;
    }

    Path getRoot() {
        return root;
    }

    private class MethodDiff {
        private String methodName;
        private String prev;
        private String updated;
        private ITree prevMethod;
        private ITree updatedMethod;

        @Contract(pure = true)
        private MethodDiff(ITree prevMethod, ITree updatedMethod) {
            this.prevMethod = prevMethod;
            this.updatedMethod = updatedMethod;
            prev = getMethodTextWithoutMethodJavadoc(prevMethod, prevFileText);
            updated = getMethodTextWithoutMethodJavadoc(updatedMethod, updatedFileText);

            String prevMethodName = getMethodName(prevMethod);
            String updatedMethodName = getMethodName(updatedMethod);
            if (!getMethodName(updatedMethod).equals(prevMethodName)) {
                System.out.println("\n" + Program.ANSI_RED + "WARNING!" + Program.ANSI_RESET);
                System.out.println("Method name in prev code does not equals to method name in updated code.");
                System.out.println("Prev method name: " + prevMethodName);
                System.out.println("Updated method name: " + updatedMethodName);
                System.out.println("Root: " + root.toAbsolutePath().toString());
            }
            methodName = prevMethodName;
        }

        @Contract(pure = true)
        private boolean hasMethodChanged() {
            return hasMethodChangedInTermsOfActions();
        }

        private boolean hasMethodChangedInTermsOfActions() {
            Set<ITree> allNodesInBothVersions = new HashSet<>();
            getAllNodes(prevMethod, allNodesInBothVersions);
            getAllNodes(updatedMethod, allNodesInBothVersions);

            Set<ITree> changedNodesInMethod = Sets.intersection(allNodesInBothVersions, changedNodes);
            return !changedNodesInMethod.isEmpty();
        }

        private void getAllNodes(ITree node, @NotNull Set<ITree> result) {
            result.add(node);
            for (ITree child: node.getChildren()) {
                getAllNodes(child, result);
            }
        }

        @NotNull
        private String getNodeText(@NotNull ITree node, @NotNull String fileContent) {
            return fileContent.substring(node.getPos(), node.getEndPos());
        }

        @NotNull
        private String getMethodTextWithoutMethodJavadoc(@NotNull ITree method, @NotNull String fileContent) {
            if (method.getChild(0).getType() == JAVADOC) {
                String textWithJavadoc = getNodeText(method, fileContent);
                ITree javadocNode = method.getChild(0);
                int endPos = javadocNode.getEndPos() - method.getPos();
                return textWithJavadoc.substring(endPos);
            } else {
                return getNodeText(method, fileContent);
            }
        }

        private String getMethodName(@NotNull ITree methodNode) {
            List<String> methodNames = methodNode.getChildren().stream()
                    .filter(child -> child.getType() == SIMPLE_NAME)
                    .map(ITree::getLabel)
                    .collect(Collectors.toList());
            if (methodNames.size() == 0) {
                System.out.println("\n" + Program.ANSI_RED + "WARNING!" + Program.ANSI_RESET);
                System.out.println("No method name found!");
                return "NoMethodNameFound_" + methodNode.hashCode();
            }
            else if (methodNames.size() > 1) {
                System.out.println("\n" + Program.ANSI_RED + "WARNING!" + Program.ANSI_RESET);
                System.out.println("More than one method names are found!");
                System.out.println("Possible names: " + String.join(", ", methodNames) + ".");
                System.out.println("First variant will be taken.");
            }
            return methodNames.get(0);
        }

        @Contract(pure = true)
        private String getMethodName() {
            return methodName;
        }

        @Contract(pure = true)
        private String getPrev() {
            return prev;
        }

        @Contract(pure = true)
        private String getUpdated() {
            return updated;
        }
    }
}
