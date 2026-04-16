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
    
    if (!data.title) {
      console.log(`⚠️ Skipping ${file}: No title found in frontmatter.`);
      continue;
    }

    const slug = generateSlug(data.title);
    const existing = await db.collection("blog-posts").where("slug", "==", slug).get();
    const html = await marked(content);

    // Payload with logic for Author and new technical fields
    const payload = {
      title: data.title,
      slug,
      excerpt: data.excerpt || "",
      
      // Author Logic: Markdown > GitHub Secret > Fallback
      authorName: data.authorName || data.author || process.env.AUTHOR_NAME || "Botmartz Team",
      authorId: data.authorId || process.env.AUTHOR_ID || "",
      authorPhotoURL: data.authorPhotoURL || "", // Optional in frontmatter
      
      category: data.category || "Technology",
      tags: data.tags || [],
      
      // Support for Soham Sir's Series Format
      series_id: data.series_id || "",
      series_slug: data.series_slug || "",
      series_title: data.series_title || "",
      difficulty: data.difficulty || "beginner",
      week: data.week || null,
      day: data.day || null,
      tools: data.tools || [],
      
      // Image Handling
      featuredImage: data.featuredImage || "",
      featuredImageAlt: data.featuredImageAlt || data.title,
      
      status: data.status || "draft",
      content: html,
      updatedAt: FieldValue.serverTimestamp(),
    };

    if (existing.empty) {
      await db.collection("blog-posts").add({
        ...payload,
        createdAt: FieldValue.serverTimestamp(),
        publishedAt: data.status === "published" ? FieldValue.serverTimestamp() : null,
        likes: 0,
        views: 0,
      });
      console.log(`✅ Created: ${data.title} by ${payload.authorName}`);
    } else {
      await existing.docs[0].ref.update(payload);
      console.log(`🔄 Updated: ${data.title} by ${payload.authorName}`);
    }
  }
  console.log("Process Completed Successfully.");
  process.exit(0);
}

publishBlogs().catch((e) => {
  console.error("❌ Error publishing blogs:", e);
  process.exit(1);
});