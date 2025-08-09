import { getProjectBySlug, getProjects } from '@/lib/data';
import { notFound } from 'next/navigation';
import Image from 'next/image';
import { Badge } from '@/components/ui/badge';
import { ArrowLeft, ArrowRight, Github, Terminal } from 'lucide-react';
import { Button } from '@/components/ui/button';
import Link from 'next/link';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { ImageGallery } from './ImageGallery';

type ProjectPageProps = {
  params: {
    slug: string;
  };
};

export default function ProjectPage({ params }: ProjectPageProps) {
  const project = getProjectBySlug(params.slug);
  
  if (!project) {
    const projectsByTag = getProjects(decodeURIComponent(params.slug));
    if (projectsByTag.length > 0) {
      return (
        <div className="container mx-auto py-12">
          <div className="mb-8">
            <Button asChild variant="outline">
              <Link href="/#projects">
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back to Projects
              </Link>
            </Button>
          </div>
          <div className="text-center">
            <h1 className="font-headline text-4xl font-bold tracking-tight md:text-5xl">
              Projects tagged: {decodeURIComponent(params.slug)}
            </h1>
          </div>
          <div className="mt-12 grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-3">
            {projectsByTag.map((p) => (
              <Link href={`/project/${p.slug}`} key={p.id} className="group flex">
                <Card className="flex w-full flex-col overflow-hidden transition-transform duration-300 group-hover:scale-105 group-hover:shadow-xl">
                  <CardHeader>
                    <div className="aspect-video overflow-hidden rounded-md">
                      <Image
                        src={p.images[0]}
                        alt={p.title}
                        data-ai-hint={p.aiHint}
                        width={600}
                        height={400}
                        className="h-full w-full object-cover"
                      />
                    </div>
                  </CardHeader>
                  <CardContent className="flex-grow">
                    <CardTitle className="font-headline text-xl">{p.title}</CardTitle>
                    <CardDescription className="mt-2">{p.description}</CardDescription>
                  </CardContent>
                  <CardFooter className="flex-col items-start gap-4">
                    <div className="flex flex-wrap gap-2">
                      {p.tags.map((tag) => (
                        <Badge key={tag} variant="outline">
                          {tag}
                        </Badge>
                      ))}
                    </div>
                    <div className="flex items-center font-medium text-primary group-hover:underline">
                      View Project <ArrowRight className="ml-2 h-4 w-4" />
                    </div>
                  </CardFooter>
                </Card>
              </Link>
            ))}
          </div>
        </div>
      );
    }
    notFound();
  }

  const otherProjects = getProjects().filter(p => p.id !== project.id);

  return (
    <>
      <article className="container mx-auto max-w-4xl py-12">
        <div className="mb-8">
          <Button asChild variant="outline">
            <Link href="/#projects">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Projects
            </Link>
          </Button>
        </div>

        <div className="text-center">
          <h1 className="font-headline text-4xl font-bold tracking-tight md:text-5xl">{project.title}</h1>
        </div>
        
        <ImageGallery project={project} />

        <div
          className="prose prose-lg mx-auto max-w-none dark:prose-invert"
          dangerouslySetInnerHTML={{ __html: project.longDescription }}
        />
        
        {project.sentimentAnalysisSection && (
          <div className="prose prose-lg mx-auto mt-8 max-w-none dark:prose-invert">
            <h3 className="font-headline text-2xl font-semibold">{project.sentimentAnalysisSection.title}</h3>
            <p>{project.sentimentAnalysisSection.content}</p>
          </div>
        )}

        {project.code && (
          <div className="mx-auto mt-8 max-w-none">
            <h3 className="flex items-center gap-2 font-headline text-2xl font-semibold">
              <Terminal className="h-6 w-6 text-primary" />
              Code Snippet
            </h3>
            <pre className="mt-4 overflow-x-auto rounded-lg bg-muted p-4 font-code text-sm text-muted-foreground">
              <code>{project.code.trim()}</code>
            </pre>
          </div>
        )}

        <div className="mt-8 text-center">
          <h3 className="font-headline text-2xl font-semibold">Technologies Used</h3>
          <div className="mt-4 flex flex-wrap justify-center gap-2">
            {project.tags.map((tag) => (
              <Link href={`/project/${encodeURIComponent(tag)}`} key={tag}>
                <Badge variant="secondary" className="text-sm">
                  {tag}
                </Badge>
              </Link>
            ))}
          </div>
        </div>

        {project.sourceCodeLink && (
          <div className="mt-8 text-center">
            <Button asChild size="lg">
              <a href={project.sourceCodeLink} target="_blank" rel="noopener noreferrer">
                <Github className="mr-2 h-5 w-5" />
                View Full Project
              </a>
            </Button>
          </div>
        )}
      </article>

      {otherProjects.length > 0 && (
        <div className="py-16">
          <div className="container mx-auto">
            <h2 className="text-center font-headline text-3xl font-bold">Other Projects</h2>
            <div className="mt-12 grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-3">
              {otherProjects.map((p) => (
                <Link href={`/project/${p.slug}`} key={p.id} className="group flex">
                  <Card className="flex w-full flex-col overflow-hidden transition-transform duration-300 group-hover:scale-105 group-hover:shadow-xl">
                    <CardHeader>
                      <div className="aspect-video overflow-hidden rounded-md">
                        <Image
                          src={p.images[0]}
                          alt={p.title}
                          data-ai-hint={p.aiHint}
                          width={600}
                          height={400}
                          className="h-full w-full object-cover"
                        />
                      </div>
                    </CardHeader>
                    <CardContent className="flex-grow">
                      <CardTitle className="font-headline text-xl">{p.title}</CardTitle>
                      <CardDescription className="mt-2">{p.description}</CardDescription>
                    </CardContent>
                    <CardFooter className="flex-col items-start gap-4">
                      <div className="flex flex-wrap gap-2">
                        {p.tags.map((tag) => (
                           <Badge variant="outline" key={tag}>{tag}</Badge>
                        ))}
                      </div>
                      <div className="flex items-center font-medium text-primary group-hover:underline">
                        View Project <ArrowRight className="ml-2 h-4 w-4" />
                      </div>
                    </CardFooter>
                  </Card>
                </Link>
              ))}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
