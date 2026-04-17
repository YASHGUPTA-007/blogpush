/**
 * One-shot local script: generate Colab notebooks for all existing blog posts,
 * inject the Colab badge into each markdown, and copy to blogs/ for publishing.
 *
 * Usage: npx tsx scripts/run-local-colab-setup.mts
 *
 * Requires: GITHUB_REPO_OWNER, GITHUB_REPO_NAME, GITHUB_BRANCH env vars
 * OR falls back to defaults for this project.
 */

import * as fs from "fs";
import * as path from "path";
import matter from "gray-matter";
import { generateNotebook } from "./generate-colab.mts";
import { injectColabLink } from "./inject-colab-link.mts";

const REPO_OWNER = process.env.GITHUB_REPO_OWNER || "YASHGUPTA-007";
const REPO_NAME  = process.env.GITHUB_REPO_NAME  || "blogpush";
const BRANCH     = process.env.GITHUB_BRANCH     || "main";

const CATEGORIES = ["pytorch", "langchain", "tensorflow", "research"];
const ROOT = process.cwd();

function updateFrontmatterColabField(mdFilePath: string, colabUrl: string): void {
  const raw = fs.readFileSync(mdFilePath, "utf-8");
  const { data, content } = matter(raw);

  if (data.colab_notebook === colabUrl) return; // already set

  data.colab_notebook = colabUrl;
  const updated = matter.stringify(content, data);
  fs.writeFileSync(mdFilePath, updated, "utf-8");
}

let total = 0;

for (const category of CATEGORIES) {
  const categoryDir = path.join(ROOT, category);
  if (!fs.existsSync(categoryDir)) continue;

  const mdFiles = fs.readdirSync(categoryDir).filter(f => f.endsWith(".md"));

  for (const mdFile of mdFiles) {
    const mdFilePath = path.join(categoryDir, mdFile);
    const basename = path.basename(mdFile, ".md");
    const notebookRelPath = `notebooks/${category}/${basename}.ipynb`;
    const notebookAbsPath = path.join(ROOT, notebookRelPath);

    const blogUrl = `https://blog.botmartz.com/${basename}`;
    const colabUrl = `https://colab.research.google.com/github/${REPO_OWNER}/${REPO_NAME}/blob/${BRANCH}/${notebookRelPath}`;

    // 1. Generate notebook
    fs.mkdirSync(path.dirname(notebookAbsPath), { recursive: true });
    const notebook = generateNotebook(mdFilePath, blogUrl);
    fs.writeFileSync(notebookAbsPath, JSON.stringify(notebook, null, 2));
    console.log(`📓 Notebook: ${notebookRelPath}`);

    // 2. Update colab_notebook field in frontmatter
    updateFrontmatterColabField(mdFilePath, colabUrl);

    // 3. Inject Colab badge into markdown
    injectColabLink(mdFilePath, colabUrl);

    // 4. Copy updated .md to blogs/
    const blogsDir = path.join(ROOT, "blogs");
    fs.mkdirSync(blogsDir, { recursive: true });
    fs.copyFileSync(mdFilePath, path.join(blogsDir, `${basename}.md`));
    console.log(`📄 Copied to blogs/${basename}.md`);

    total++;
  }
}

console.log(`\n✅ Done. Processed ${total} blog posts.`);
console.log(`\nNext steps:`);
console.log(`  1. git add notebooks/ blogs/ pytorch/ langchain/ tensorflow/ research/`);
console.log(`  2. git commit -m "feat: add Colab notebooks and inject badges"`);
console.log(`  3. git push`);
console.log(`  4. npm run publish-blogs  (to update Firestore)`);
