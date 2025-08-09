import type { BlogPost } from '@/lib/data';
import Link from 'next/link';
import Image from 'next/image';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { ArrowRight } from 'lucide-react';

type BlogSectionProps = {
  posts: BlogPost[];
};

export function BlogSection({ posts }: BlogSectionProps) {
  return (
    <section id="blog" className="container mx-auto py-16">
      <div className="text-center">
        <h2 className="font-headline text-3xl font-bold">From the Blog</h2>
        <p className="mx-auto mt-4 max-w-2xl text-muted-foreground">
          I share my thoughts on data science, technology, and career development. Here are some of my latest articles.
        </p>
      </div>

      <div className="mt-12 grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-3">
        {posts.map((post) => (
          <Card key={post.id} className="flex flex-col overflow-hidden transition-transform duration-300 hover:scale-105 hover:shadow-xl">
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
            <CardContent className="flex-grow p-6">
              <h3 className="font-headline text-xl font-semibold">
                <Link href={`/blog/${post.slug}`} className="hover:text-primary">
                  {post.title}
                </Link>
              </h3>
              <p className="mt-2 text-sm text-muted-foreground">{post.summary}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="mt-12 text-center">
        <Button asChild variant="outline" size="lg">
          <Link href="/blog">
            View All Posts <ArrowRight className="ml-2 h-4 w-4" />
          </Link>
        </Button>
      </div>
    </section>
  );
}
