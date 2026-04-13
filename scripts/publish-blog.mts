import { initializeApp, cert, getApps } from "firebase-admin/app";
import { getFirestore, FieldValue } from "firebase-admin/firestore";
import * as fs from "fs";
import * as path from "path";
import matter from "gray-matter";
import { marked } from "marked";
import * as dotenv from "dotenv";

dotenv.config({ path: ".env.local" });

const app = getApps().length > 0 ? getApps()[0] : initializeApp({
  credential: cert({
    projectId: process.env.FIREBASE_ADMIN_PROJECT_ID,
    clientEmail: process.env.FIREBASE_ADMIN_CLIENT_EMAIL,
    privateKey: process.env.FIREBASE_ADMIN_PRIVATE_KEY?.replace(/\\n/g, "\n"),
  }),
});

const db = getFirestore(app);

function generateSlug(title: string): string {
  return title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

async function publishBlogs() {
  const blogsDir = path.join(process.cwd(), "blogs");
  if (!fs.existsSync(blogsDir)) {
      console.log("No blogs directory found.");
      process.exit(0);
  }
  
  const files = fs.readdirSync(blogsDir).filter((f) => f.endsWith(".md"));

  for (const file of files) {
    const raw = fs.readFileSync(path.join(blogsDir, file), "utf-8");
    const { data, content } = matter(raw);
    const slug = generateSlug(data.title);

    const existing = await db.collection("blog-posts").where("slug", "==", slug).get();
    const html = await marked(content);

    const payload = {
      title: data.title,
      slug,
      excerpt: data.excerpt || "",
      author: data.author || "",
      category: data.category || "Technology",
      tags: data.tags || [],
      status: data.status || "draft",
      content: html,
      updatedAt: FieldValue.serverTimestamp(),
    };

    if (existing.empty) {
      await db.collection("blog-posts").add({
        ...payload,
        createdAt: FieldValue.serverTimestamp(),
        publishedAt: data.status === "published" ? FieldValue.serverTimestamp() : null,
      });
      console.log(`✅ Created: ${data.title}`);
    } else {
      await existing.docs[0].ref.update(payload);
      console.log(`🔄 Updated: ${data.title}`);
    }
  }
  console.log("Done.");
  process.exit(0);
}

publishBlogs().catch((e) => { console.error(e); process.exit(1); });