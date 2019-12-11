import com.github.gumtreediff.client.Run;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

public class Program {
    public static void main(@NotNull String[] args) {
        printHelp();
        if (args.length != 1) {
            System.out.println("Number of input arguments must be 1");
            return;
        }

        Run.initGenerators();

        Path root = Paths.get(args[0]);
        final int[] numberOfFiles = {0, 0}; // [failed to parse, total]
        try {
            List<DiffExtractor> extractors = Files.walk(root).filter(Files::isDirectory).map(dirRoot -> {
                DiffExtractor diffExtractor = null;
                try {
                    diffExtractor = new DiffExtractor(dirRoot);
                } catch (NoSuchFileException ignored) {
                    return null;
                } catch (IOException e) {
                    System.err.println("WARNING!");
                    System.out.println("Exception occurred during diffing of two files");
                    System.out.println("Exception: " + e.getMessage());
                    numberOfFiles[0] += 1;
                }
                numberOfFiles[1] += 1;
                return diffExtractor;
            }).filter(Objects::nonNull).collect(Collectors.toList());
            for (DiffExtractor extractor: extractors) {
                try {
                    extractor.saveMethodDiffs();
                } catch (IOException e) {
                    System.out.println("Program cannot create a directory or write a file or read a source file!");
                    System.out.println("Exception: " + e.getMessage());
                    e.printStackTrace();
                }
            }
        } catch (IOException e) {
            System.out.println("Passed path does not exist!");
            System.out.println("Exception: " + e.getMessage());
            e.printStackTrace();
        }
        System.out.println("Number of files failed to parse: " +
                numberOfFiles[0] +" / " + numberOfFiles[1] + " = " + ((double) numberOfFiles[0] / numberOfFiles[1]));
    }

    private static void printHelp() {
        System.out.println("Usage: <root path to data>\n" +
                "Under this folder recursively all pairs of prev and updated files will be parsed and proccessed.\n");
    }
}
