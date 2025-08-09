'use client';

import { useState } from 'react';
import Image from 'next/image';
import { cn } from '@/lib/utils';
import type { Project } from '@/lib/data';

type ImageGalleryProps = {
  project: Project;
};

export function ImageGallery({ project }: ImageGalleryProps) {
  const [activeImage, setActiveImage] = useState(project.images[0]);

  return (
    <div className="my-8">
      <div className="aspect-video overflow-hidden rounded-lg">
        <Image
          src={activeImage}
          alt={project.title}
          data-ai-hint={project.aiHint}
          width={1200}
          height={600}
          className="h-full w-full object-cover"
          priority
        />
      </div>
      {project.images.length > 1 && (
        <div className="mt-4 grid grid-cols-4 gap-4">
          {project.images.map((image, index) => (
            <div
              key={index}
              onClick={() => setActiveImage(image)}
              className={cn(
                'cursor-pointer overflow-hidden rounded-md border-2 transition-all',
                activeImage === image
                  ? 'border-primary'
                  : 'border-transparent hover:border-primary/50'
              )}
            >
              <Image
                src={image}
                alt={`${project.title} thumbnail ${index + 1}`}
                data-ai-hint={project.aiHint}
                width={300}
                height={200}
                className="h-full w-full object-cover"
              />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
