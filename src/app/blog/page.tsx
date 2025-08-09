import { getBlogPosts } from '@/lib/data';
import Image from 'next/image';
import Link from 'next/link';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Calendar } from 'lucide-react';

export default function BlogPage() {
  const posts = getBlogPosts();

  return (
    <div className="container mx-auto py-12">
      <div className="text-center">
        <h1 className="font-headline text-4xl font-bold tracking-tight">Blog</h1>
        <p className="mt-4 max-w-2xl mx-auto text-lg text-muted-foreground">
          Insights on data science, machine learning, and technology.
        </p>
      </div>

      <div className="mt-12 grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-3">
        {posts.map((post) => (
          <Card key={post.id} className="flex flex-col overflow-hidden transition-shadow duration-300 hover:shadow-xl">
            <CardHeader className="p-0">
              <Link href={`/blog/${post.slug}`}>
                <div className="aspect-video">
                  <Image
                    src={post.image}
                    alt={post.title}
                    data-ai-hint={post.aiHint}
                    width={800}
                    height={400}
                    className="h-full w-full object-cover"
                  />
                </div>
              </Link>
            </CardHeader>
            <CardContent className="flex flex-grow flex-col p-6">
              <div className="flex-grow">
                <h2 className="font-headline text-xl font-semibold">
                  <Link href={`/blog/${post.slug}`} className="hover:text-primary">
                    {post.title}
                  </Link>
                </h2>
                <p className="mt-2 text-sm text-muted-foreground">{post.summary}</p>
              </div>
              <div className="mt-4 flex items-center text-xs text-muted-foreground">
                <Calendar className="mr-1.5 h-4 w-4" />
                <span>{new Date(post.date).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}</span>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
