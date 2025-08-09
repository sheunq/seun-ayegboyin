import { getBlogPostBySlug, getBlogPosts } from '@/lib/data';
import { notFound } from 'next/navigation';
import Image from 'next/image';
import { Calendar, User } from 'lucide-react';

type BlogPostPageProps = {
  params: {
    slug: string;
  };
};

// This function allows Next.js to generate static pages for each blog post at build time.
export async function generateStaticParams() {
  const posts = getBlogPosts();
  return posts.map((post) => ({
    slug: post.slug,
  }));
}

export default function BlogPostPage({ params }: BlogPostPageProps) {
  const post = getBlogPostBySlug(params.slug);

  if (!post) {
    notFound();
  }

  return (
    <article className="container mx-auto max-w-4xl py-12">
      <div className="text-center">
        <h1 className="font-headline text-4xl font-bold tracking-tight md:text-5xl">{post.title}</h1>
        <div className="mt-4 flex justify-center gap-6 text-sm text-muted-foreground">
          <div className="flex items-center gap-1.5">
            <User className="h-4 w-4" />
            <span>{post.author}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <Calendar className="h-4 w-4" />
            <span>{new Date(post.date).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}</span>
          </div>
        </div>
      </div>
      
      <div className="my-8 aspect-video overflow-hidden rounded-lg">
        <Image
          src={post.image}
          alt={post.title}
          data-ai-hint={post.aiHint}
          width={1200}
          height={600}
          className="h-full w-full object-cover"
          priority
        />
      </div>

      <div
        className="prose prose-lg mx-auto max-w-none dark:prose-invert"
        dangerouslySetInnerHTML={{ __html: post.content }}
      />
    </article>
  );
}
