
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import Image from 'next/image';
import Link from 'next/link';
import type { ProjectCategory } from '@/lib/data';
import { ArrowRight } from 'lucide-react';

type ProjectsSectionProps = {
  categories: ProjectCategory[];
};

export function ProjectsSection({ categories }: ProjectsSectionProps) {
  return (
    <section id="projects" className="container mx-auto py-16">
      <div className="text-center">
        <h2 className="font-headline text-3xl font-bold">Project Categories</h2>
        <p className="mx-auto mt-4 max-w-2xl text-muted-foreground">
          Explore projects based on different categories. Each category showcases my ability to handle data, apply algorithms, and deliver impactful results.
        </p>
      </div>

      <div className="mt-12 grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-3">
        {categories.map((category) => (
          <Link href={`/project/${category.slug}`} key={category.id} className="group flex">
            <Card className="flex w-full flex-col overflow-hidden transition-transform duration-300 group-hover:scale-105 group-hover:shadow-xl">
              <CardHeader>
                <div className="aspect-video overflow-hidden rounded-md">
                  <Image
                    src={category.image}
                    alt={category.title}
                    data-ai-hint={category.aiHint}
                    width={600}
                    height={400}
                    className="h-full w-full object-cover"
                  />
                </div>
              </CardHeader>
              <CardContent className="flex-grow">
                <CardTitle className="font-headline text-xl">{category.title}</CardTitle>
                <CardDescription className="mt-2">{category.description}</CardDescription>
              </CardContent>
              <CardFooter>
                 <div className="flex items-center font-medium text-primary group-hover:underline">
                  View Projects <ArrowRight className="ml-2 h-4 w-4" />
                </div>
              </CardFooter>
            </Card>
          </Link>
        ))}
      </div>
    </section>
  );
}
