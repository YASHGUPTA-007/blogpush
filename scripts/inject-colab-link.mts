import * as fs from "fs";

const BADGE_PREFIX = "colab-badge.svg";

export function injectColabLink(mdFilePath: string, colabUrl: string): void {
  let content = fs.readFileSync(mdFilePath, "utf-8");

  // Idempotent: skip if badge already present
  if (content.includes(BADGE_PREFIX)) {
    console.log(`⏭️  Colab badge already present: ${mdFilePath}`);
    return;
  }

  const badge = `<a href="${colabUrl}" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="height:28px;margin-bottom:1rem;" /></a>`;

  // Insert after closing frontmatter ---
  const frontmatterEnd = content.indexOf("---", 3);
  if (frontmatterEnd === -1) {
    console.warn(`⚠️  No frontmatter found in ${mdFilePath}, prepending badge.`);
    content = badge + "\n\n" + content;
  } else {
    const insertAt = frontmatterEnd + 3;
    content = content.slice(0, insertAt) + "\n\n" + badge + "\n" + content.slice(insertAt);
  }

  fs.writeFileSync(mdFilePath, content, "utf-8");
  console.log(`✅ Colab badge injected: ${mdFilePath}`);
}

// CLI: node inject-colab-link.mts <md-file> <colab-url>
if (process.argv[2]) {
  const [, , mdFile, colabUrl] = process.argv;
  if (!colabUrl) {
    console.error("Usage: inject-colab-link.mts <md-file> <colab-url>");
    process.exit(1);
  }
  injectColabLink(mdFile, colabUrl);
}
