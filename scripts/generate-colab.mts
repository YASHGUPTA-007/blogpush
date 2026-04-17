import * as fs from "fs";
import * as path from "path";
import matter from "gray-matter";

const INSTALL_MAP: Record<string, string> = {
  pytorch: "!pip install -q torch torchvision torchaudio",
  langchain: "!pip install -q langchain langchain-openai langchain-community chromadb",
  tensorflow: "!pip install -q tensorflow mlflow",
  research: "!pip install -q torch transformers peft bitsandbytes",
  "genai-news": "",
};

function extractPythonBlocks(content: string): string[] {
  const blocks: string[] = [];
  const regex = /```python\n([\s\S]*?)```/g;
  let match;
  while ((match = regex.exec(content)) !== null) {
    blocks.push(match[1].trimEnd());
  }
  return blocks;
}

function makeCell(type: "markdown" | "code", source: string) {
  return {
    cell_type: type,
    metadata: {},
    source: source.split("\n").map((line, i, arr) => (i < arr.length - 1 ? line + "\n" : line)),
    ...(type === "code" ? { outputs: [], execution_count: null } : {}),
  };
}

export function generateNotebook(mdFilePath: string, blogUrl: string): object {
  const raw = fs.readFileSync(mdFilePath, "utf-8");
  const { data, content } = matter(raw);

  const title = data.title || path.basename(mdFilePath, ".md");
  // Use directory name (pytorch/langchain/etc.) not frontmatter category ("AI")
  const dirName = path.basename(path.dirname(path.resolve(mdFilePath)));
  const installCmd = INSTALL_MAP[dirName] ?? "";

  const cells: object[] = [];

  // Header cell
  cells.push(
    makeCell(
      "markdown",
      `# ${title}\n\n> This notebook contains all code examples from the blog post.\n> [Read the full post on BotMartz](${blogUrl})\n\n**Author:** Soham Sharma · blog.botmartz.com`
    )
  );

  // Install cell (if applicable)
  if (installCmd) {
    cells.push(makeCell("code", installCmd));
  }

  // One code cell per Python block
  const blocks = extractPythonBlocks(content);
  for (const block of blocks) {
    cells.push(makeCell("code", block));
  }

  return {
    nbformat: 4,
    nbformat_minor: 5,
    metadata: {
      kernelspec: {
        display_name: "Python 3",
        language: "python",
        name: "python3",
      },
      language_info: {
        name: "python",
        version: "3.10.0",
      },
      colab: {
        name: `${path.basename(mdFilePath, ".md")}.ipynb`,
        provenance: [],
      },
    },
    cells,
  };
}

// CLI: node generate-colab.mts <md-file> <blog-url> <output-path>
if (process.argv[2]) {
  const [, , mdFile, blogUrl = "https://blog.botmartz.com", outFile] = process.argv;
  const notebook = generateNotebook(mdFile, blogUrl);
  const dest = outFile || mdFile.replace(/\.md$/, ".ipynb");
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.writeFileSync(dest, JSON.stringify(notebook, null, 2));
  console.log(`✅ Notebook written: ${dest}`);
}
